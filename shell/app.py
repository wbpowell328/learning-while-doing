import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import apps
from apps.cash_balance import SimConfig, simulate  # 1-D default for legacy batch/reveal endpoints
from policy import (
    BeliefConfig, AcquisitionConfig, SessionConfig,
    RandomPolicy, IEPolicy, KGPolicy, Session,
    KGMCPolicy, KGIndependentPolicy, OKGCorrelatedPolicy, OKGIndependentPolicy,
    kg_analytic_correlated_at, kg_mc_correlated_at, kg_independent_at,
)
from policy.acquire import _make_grid
from .models import (
    CreateSessionRequest, CreateSessionResponse,
    StepResponse, JumpEventOut,
    EvaluateRequest,
    ObserveRequest, ObserveResponse,
    StateResponse, PosteriorResponse, Posterior2DResponse, KG2DResponse,
    RevealResponse, KGComparisonResponse,
    BatchRequest, BatchResponse, BatchPolicyResult,
)


def _as_list(x) -> list[float]:
    """Coerce scalar/ndarray/list into a plain Python list of floats."""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    return [float(v) for v in arr.ravel()]


def _theta_out(theta, dim: int):
    """Format θ for JSON responses: scalar for dim=1, list for dim>=2."""
    if dim == 1:
        return float(np.atleast_1d(np.asarray(theta, dtype=float))[0])
    return _as_list(theta)

app = FastAPI(title="learning-while-doing")

# CORS: default to Vite dev; override in prod via ALLOWED_ORIGINS (comma-separated).
_allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> Session
_sessions: dict[str, Session] = {}


def _make_step_response(result, session: Session) -> StepResponse:
    dim = session.dim
    payload = dict(
        impparam=_theta_out(result.impparam, dim),
        total_cost=float(result.total_cost),
        opportunity_cost=float(result.opportunity_cost),
        # `shortfall_cost` — for cash_balance_2d this is a computed property
        # summing the ind + inst components; both apps expose the attribute.
        shortfall_cost=float(getattr(result, "shortfall_cost", 0.0)),
        days=int(result.days),
        n_steps=int(session.n_steps),
        best_impparam=_theta_out(session.best_impparam(), dim),
        initial_aum=float(session._sim_config.initial_aum),
    )
    # 1-D app carries a cash_series and event_log; expose via the extras.
    if hasattr(result, "cash_series"):
        payload["cash_series"] = result.cash_series.tolist()
    if hasattr(result, "event_log"):
        payload["event_log"] = [
            JumpEventOut(day=e.day, size_fraction=e.size_fraction, direction=e.direction).model_dump()
            for e in result.event_log
        ]
    return StepResponse(**payload)


def _make_policy(name: str, acq_cfg: AcquisitionConfig, budget: int | None):
    if name == "ie":
        return IEPolicy(acq_cfg)
    if name == "kg":
        return KGPolicy(acq_cfg)                # offline correlated (analytic)
    if name == "kg_indep":
        return KGIndependentPolicy(acq_cfg)     # offline independent
    if name == "okg":
        return OKGCorrelatedPolicy(acq_cfg, budget=budget or 10)
    if name == "okg_indep":
        return OKGIndependentPolicy(acq_cfg, budget=budget or 10)
    return RandomPolicy(acq_cfg)  # "random" and "human" both use RandomPolicy


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/sessions", status_code=201)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    # Look up the requested application in the registry.
    try:
        app_mod = apps.get_app(req.app_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build per-app SimConfig. Only carry over fields the client explicitly set
    # (so we don't overwrite the app's 2-D tuple defaults with 1-D scalar defaults
    # from the SimConfigIn schema).  Also honor any extra fields (e.g. the 2-D
    # config's initial_aum_ind_fraction) via the model's __pydantic_extra__.
    sim_cfg_kwargs = req.sim_config.model_dump(exclude_unset=True)
    sim_field_names = {f for f in app_mod.SimConfig.__dataclass_fields__}
    sim_cfg = app_mod.SimConfig(**{k: v for k, v in sim_cfg_kwargs.items() if k in sim_field_names})

    # Take θ bounds from the sim config (source of truth for the box).
    lo = _as_list(sim_cfg.impparam_min)
    hi = _as_list(sim_cfg.impparam_max)
    dim = int(app_mod.THETA_DIM)

    # Build belief config — start with client-provided fields only, then
    # broadcast a scalar length_scale to per-dim if the app is multi-D and the
    # client didn't already provide a per-dim scale.
    belief_kwargs = req.belief_config.model_dump(exclude_unset=True)
    if dim > 1:
        ls = belief_kwargs.get("length_scale")
        if ls is None or isinstance(ls, (int, float)):
            # Fall back to BeliefConfig default (0.04) if client didn't set one.
            base = float(ls) if isinstance(ls, (int, float)) else 0.04
            belief_kwargs["length_scale"] = [base] * dim
    bel_field_names = {f for f in BeliefConfig.__dataclass_fields__}
    belief_cfg = BeliefConfig(**{k: v for k, v in belief_kwargs.items() if k in bel_field_names})

    # Acquisition config — bounds always come from the SimConfig (source of truth).
    acq_kwargs = req.acq_config.model_dump(exclude_unset=True)
    acq_kwargs["impparam_min"] = lo if dim > 1 else lo[0]
    acq_kwargs["impparam_max"] = hi if dim > 1 else hi[0]
    # Cap grid_size for higher dims to keep O(grid_size**dim) manageable.
    if dim >= 2 and acq_kwargs.get("grid_size", 100) >= 50:
        acq_kwargs["grid_size"] = 25
    acq_field_names = {f for f in AcquisitionConfig.__dataclass_fields__}
    acq_cfg = AcquisitionConfig(**{k: v for k, v in acq_kwargs.items() if k in acq_field_names})

    ses_cfg = SessionConfig(**req.session_config.model_dump())
    policy = _make_policy(req.policy, acq_cfg, req.budget)

    session = Session(
        sim_config=sim_cfg,
        belief_config=belief_cfg,
        acq_config=acq_cfg,
        session_config=ses_cfg,
        policy=policy,
        session_seed=req.session_seed,
        simulate_fn=app_mod.simulate,
    )
    sid = str(uuid4())
    _sessions[sid] = session

    return CreateSessionResponse(
        session_id=sid,
        policy=req.policy,
        app_name=req.app_name,
        dim=dim,
        minimize=bool(app_mod.MINIMIZE),
        impparam_min=lo,
        impparam_max=hi,
    )


@app.post("/sessions/{sid}/step")
def step(sid: str) -> StepResponse:
    session = _get_or_404(sid)
    result = session.step()
    return _make_step_response(result, session)


@app.post("/sessions/{sid}/observe")
def observe(sid: str, body: ObserveRequest) -> ObserveResponse:
    session = _get_or_404(sid)
    session.observe(body.impparam, body.total_cost)
    return ObserveResponse(
        n_observations=session.belief.n_observations,
        best_impparam=_theta_out(session.best_impparam(), session.dim),
    )


@app.get("/sessions/{sid}/state")
def state(sid: str) -> StateResponse:
    session = _get_or_404(sid)
    dim = session.dim
    hist = [(_theta_out(t, dim), float(c)) for t, c in session.history]
    return StateResponse(
        n_steps=session.n_steps,
        n_observations=session.belief.n_observations,
        best_impparam=_theta_out(session.best_impparam(), dim),
        history=hist,
    )


@app.post("/sessions/{sid}/evaluate")
def evaluate(sid: str, body: EvaluateRequest) -> StepResponse:
    session = _get_or_404(sid)
    result = session.evaluate(body.impparam)
    return _make_step_response(result, session)


@app.get("/sessions/{sid}/posterior_2d")
def posterior_2d(sid: str, grid_size: int = 30) -> Posterior2DResponse:
    """
    Return the 2-D GP posterior surface (mean + std) on a grid_size × grid_size
    grid over the θ box, plus the observation history in flat form.  For 2-D
    apps only; raises 400 on 1-D sessions.
    """
    session = _get_or_404(sid)
    if session.dim != 2:
        raise HTTPException(status_code=400, detail=f"posterior_2d requires a 2-D session; this one has dim={session.dim}")

    cfg = session._acq_config
    lo = _as_list(cfg.impparam_min)
    hi = _as_list(cfg.impparam_max)

    axis1 = np.linspace(lo[0], hi[0], grid_size)
    axis2 = np.linspace(lo[1], hi[1], grid_size)
    G1, G2 = np.meshgrid(axis1, axis2, indexing="ij")
    grid = np.stack([G1.ravel(), G2.ravel()], axis=-1)   # (grid_size**2, 2)

    mean, std = session.belief.posterior(grid)

    # History flattened as rows [theta1, theta2, cost]
    hist_rows: list[list[float]] = []
    for t, c in session.history:
        arr = np.atleast_1d(np.asarray(t, dtype=float))
        hist_rows.append([float(arr[0]), float(arr[1]), float(c)])

    best = _as_list(session.best_impparam())
    return Posterior2DResponse(
        axis1=axis1.tolist(),
        axis2=axis2.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        history=hist_rows,
        best_impparam=best,
    )


@app.get("/sessions/{sid}/kg_2d")
def kg_2d(sid: str, grid_size: int = 20) -> KG2DResponse:
    """
    Return the analytic correlated-KG surface KG(θ) at every point on a
    grid_size × grid_size grid.  For 2-D apps only.
    """
    session = _get_or_404(sid)
    if session.dim != 2:
        raise HTTPException(status_code=400,
            detail=f"kg_2d requires a 2-D session; this one has dim={session.dim}")

    cfg = session._acq_config
    lo = _as_list(cfg.impparam_min)
    hi = _as_list(cfg.impparam_max)

    axis1 = np.linspace(lo[0], hi[0], grid_size)
    axis2 = np.linspace(lo[1], hi[1], grid_size)
    G1, G2 = np.meshgrid(axis1, axis2, indexing="ij")
    grid = np.stack([G1.ravel(), G2.ravel()], axis=-1)   # (grid_size**2, 2)

    # candidates == grid — KG at every grid point using itself as the search set.
    kg = kg_analytic_correlated_at(session.belief, grid, grid)

    hist_rows: list[list[float]] = []
    for t, c in session.history:
        arr = np.atleast_1d(np.asarray(t, dtype=float))
        hist_rows.append([float(arr[0]), float(arr[1]), float(c)])

    best = _as_list(session.best_impparam())
    return KG2DResponse(
        axis1=axis1.tolist(),
        axis2=axis2.tolist(),
        kg=kg.tolist(),
        history=hist_rows,
        best_impparam=best,
    )


@app.get("/sessions/{sid}/reveal")
def reveal(sid: str, grid_size: int = 30, n_reps: int = 12) -> RevealResponse:
    session = _get_or_404(sid)
    cfg = session._sim_config
    sc = session._sc
    base_seed = session._session_seed + 999_000

    grid = np.linspace(cfg.impparam_min, cfg.impparam_max, grid_size)
    mean_costs: list[float] = []
    for c in grid:
        costs = [
            simulate(
                config=cfg,
                impparam=float(c),
                horizon_weeks=sc.horizon_weeks,
                session_seed=base_seed,
                experiment_index=i,
            ).total_cost
            for i in range(n_reps)
        ]
        mean_costs.append(float(np.mean(costs)))

    true_best_idx = int(np.argmin(mean_costs))
    player_c = session.best_impparam()
    player_idx = int(np.argmin(np.abs(grid - player_c)))

    naive_c = float(np.clip(0.10, cfg.impparam_min, cfg.impparam_max))
    naive_costs = [
        simulate(
            config=cfg,
            impparam=naive_c,
            horizon_weeks=sc.horizon_weeks,
            session_seed=base_seed,
            experiment_index=i,
        ).total_cost
        for i in range(n_reps)
    ]

    return RevealResponse(
        impparams=grid.tolist(),
        mean_cost=mean_costs,
        true_best_impparam=float(grid[true_best_idx]),
        true_min_cost=mean_costs[true_best_idx],
        player_best_impparam=player_c,
        player_best_cost=mean_costs[player_idx],
        naive_cost=float(np.mean(naive_costs)),
    )


@app.post("/sessions/batch")
def batch_run(req: BatchRequest):
    """
    Streams NDJSON: one JSON object per line, terminated by \\n.

    Events:
      {"type": "started", "total_policies": N, "total_sims_per_policy": S,
       "total_runs": N*S}
      {"type": "progress", "completed": k, "total": N*S,
       "current_policy": "...", "sim_idx": i}   (emitted after each run)
      {"type": "ground_truth", "true_best_impparam": ..., "true_min_cost": ...}
      {"type": "result", ... full BatchResponse fields ...}

    Common Random Numbers: policies within the same sim_idx share a seed,
    so cost differences reflect policy behavior rather than noise draws.
    """
    sim_cfg    = SimConfig(**req.sim_config.model_dump())
    belief_cfg = BeliefConfig(**req.belief_config.model_dump())
    acq_cfg    = AcquisitionConfig(**req.acq_config.model_dump())
    ses_cfg    = SessionConfig(**req.session_config.model_dump())

    # Build (label, param, factory) for each policy in the family
    if req.family == "KG":
        family = [
            ("KG offline correlated (analytic)", 0.0, lambda: KGPolicy(acq_cfg)),
            ("KG offline correlated (MC, n=50)", 1.0, lambda: KGMCPolicy(acq_cfg, n_mc=50)),
            ("KG offline independent", 2.0, lambda: KGIndependentPolicy(acq_cfg)),
            ("KG online correlated (Ryzhov)", 3.0, lambda: OKGCorrelatedPolicy(acq_cfg, budget=req.budget)),
            ("KG online independent", 4.0, lambda: OKGIndependentPolicy(acq_cfg, budget=req.budget)),
        ]
    elif req.family == "IE":
        family = []
        for k in range(21):
            z = round(k * 0.2, 3)
            cfg_z = AcquisitionConfig(
                impparam_min=acq_cfg.impparam_min, impparam_max=acq_cfg.impparam_max,
                grid_size=acq_cfg.grid_size, z_alpha=z,
            )
            family.append((f"IE (z={z:.2f})", z, (lambda c=cfg_z: IEPolicy(c))))
    else:
        raise HTTPException(400, f"unknown family: {req.family}")

    total_runs = len(family) * req.sims_per_policy

    def stream():
        yield json.dumps({
            "type": "started",
            "family": req.family,
            "total_policies": len(family),
            "total_sims_per_policy": req.sims_per_policy,
            "total_runs": total_runs,
            "budget": req.budget,
        }) + "\n"

        per_policy: dict[str, dict] = {
            label: {"param": param, "best": [], "term": [], "cum": []}
            for label, param, _ in family
        }

        completed = 0
        for sim_idx in range(req.sims_per_policy):
            base_seed = req.session_seed + sim_idx * 10_000
            for label, _, factory in family:
                session = Session(sim_cfg, belief_cfg, acq_cfg, ses_cfg, factory(), base_seed)
                cumulative = 0.0
                for _ in range(req.budget):
                    result = session.step()
                    cumulative += result.total_cost
                best = session.best_impparam()
                term = _evaluate_expected_cost(sim_cfg, ses_cfg, best,
                                                base_seed + 999_000, n_reps=12)
                per_policy[label]["best"].append(best)
                per_policy[label]["term"].append(term)
                per_policy[label]["cum"].append(cumulative)

                completed += 1
                yield json.dumps({
                    "type": "progress",
                    "completed": completed,
                    "total": total_runs,
                    "current_policy": label,
                    "sim_idx": sim_idx + 1,
                }) + "\n"

        # Ground truth (once, after all sims done)
        true_best, true_min = _ground_truth(
            sim_cfg, ses_cfg, acq_cfg,
            base_seed_prefix=req.session_seed + 777_000, grid_size=30, n_reps=12,
        )
        yield json.dumps({
            "type": "ground_truth",
            "true_best_impparam": true_best,
            "true_min_cost": true_min,
        }) + "\n"

        # Aggregate
        def _agg(vals):
            arr = np.array(vals)
            return float(arr.mean()), float(arr.std())

        results = []
        for label, param, _ in family:
            d = per_policy[label]
            m_best, s_best = _agg(d["best"])
            m_term, s_term = _agg(d["term"])
            m_cum, s_cum = _agg(d["cum"])
            results.append({
                "policy": label, "param": param,
                "mean_best_impparam": m_best, "std_best_impparam": s_best,
                "mean_terminal_cost": m_term, "std_terminal_cost": s_term,
                "mean_cumulative_cost": m_cum, "std_cumulative_cost": s_cum,
                "best_impparams": d["best"],
                "terminal_costs": d["term"],
                "cumulative_costs": d["cum"],
            })

        yield json.dumps({
            "type": "result",
            "family": req.family,
            "sims_per_policy": req.sims_per_policy,
            "budget": req.budget,
            "session_seed": req.session_seed,
            "true_best_impparam": true_best,
            "true_min_cost": true_min,
            "policies": results,
        }) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        # Disable proxy buffering so events reach the client as they're emitted.
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


def _evaluate_expected_cost(sim_cfg: SimConfig, ses_cfg: SessionConfig,
                             impparam: float, base_seed: int, n_reps: int = 12) -> float:
    costs = [
        simulate(config=sim_cfg, impparam=float(impparam),
                 horizon_weeks=ses_cfg.horizon_weeks,
                 session_seed=base_seed, experiment_index=i).total_cost
        for i in range(n_reps)
    ]
    return float(np.mean(costs))


def _ground_truth(sim_cfg: SimConfig, ses_cfg: SessionConfig, acq_cfg: AcquisitionConfig,
                   base_seed_prefix: int, grid_size: int = 30, n_reps: int = 12) -> tuple[float, float]:
    grid = np.linspace(acq_cfg.impparam_min, acq_cfg.impparam_max, grid_size)
    mean_costs = [
        _evaluate_expected_cost(sim_cfg, ses_cfg, float(c), base_seed_prefix, n_reps)
        for c in grid
    ]
    i = int(np.argmin(mean_costs))
    return float(grid[i]), float(mean_costs[i])


@app.get("/sessions/{sid}/kg")
def kg_comparison(
    sid: str,
    spacing: float = 0.05,
    mc_samples: int = 50,
    mc_seed: int = 12345,
    budget: int = 10,
) -> KGComparisonResponse:
    """
    KG at a coarse probe grid (default 5% spacing across [impparam_min, impparam_max]).

    Three offline KG series (value-of-information only):
      - analytic_correlated: exact FPD closed form (the one the policy uses)
      - mc_correlated:       Monte-Carlo estimate of the same quantity
      - independent:         closed form assuming zero cross-covariance

    Two online KG series (Ryzhov 2010, min-cost form):
      online_KG(x) = mu_n(x) - (N - n) * offline_KG(x)
    where N = `budget`, n = steps taken so far. Choose x* = argmin online_KG.
    """
    session = _get_or_404(sid)
    cfg = session._acq_config

    # Build probe grid: multiples of `spacing` inside [impparam_min, impparam_max].
    first = float(np.ceil(cfg.impparam_min / spacing) * spacing)
    if first < cfg.impparam_min - 1e-9:
        first += spacing
    probes = np.arange(first, cfg.impparam_max + spacing * 1e-6, spacing)
    if probes.size == 0:
        probes = np.array([0.5 * (cfg.impparam_min + cfg.impparam_max)])

    # Search grid: the same 100-point grid the KG policy uses.
    search_grid = np.linspace(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    ana = kg_analytic_correlated_at(session.belief, search_grid, probes)
    mc = kg_mc_correlated_at(
        session.belief, search_grid, probes,
        n_mc=mc_samples, rng=np.random.default_rng(mc_seed),
    )
    ind = kg_independent_at(session.belief, search_grid, probes)

    # Posterior mean at the probe points — needed for the online KG composite.
    mu_probes, _ = session.belief.posterior(probes)

    # Online KG (Ryzhov, min-cost form). Non-negative multiplier so late in
    # the run (n → N) the info-value bonus disappears and OKG reduces to μ_n(x).
    steps_used = int(session.n_steps)
    remaining = max(0, int(budget) - steps_used)
    online_ana = (mu_probes - remaining * ana).tolist()
    online_ind = (mu_probes - remaining * ind).tolist()

    return KGComparisonResponse(
        impparams=probes.tolist(),
        posterior_mean=mu_probes.tolist(),
        analytic_correlated=ana.tolist(),
        mc_correlated=mc.tolist(),
        independent=ind.tolist(),
        online_correlated=online_ana,
        online_independent=online_ind,
        budget=int(budget),
        steps_used=steps_used,
        mc_samples=mc_samples,
        mc_seed=mc_seed,
    )


@app.get("/sessions/{sid}/posterior")
def posterior(sid: str, grid_size: int = 200) -> PosteriorResponse:
    session = _get_or_404(sid)
    cfg = session._acq_config
    grid = np.linspace(cfg.impparam_min, cfg.impparam_max, grid_size)
    mean, std = session.belief.posterior(grid)
    return PosteriorResponse(
        impparams=grid.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        best_impparam=session.best_impparam(),
    )


@app.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: str) -> None:
    _get_or_404(sid)
    del _sessions[sid]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(sid: str) -> Session:
    session = _sessions.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {sid!r} not found")
    return session


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
# In production the build step runs `npm run build` in frontend/, producing
# frontend/dist/. We mount that here so a single Render service serves both
# the API and the UI at the same origin (no CORS, one URL).
# MUST be mounted LAST so all @app.get/@app.post routes take precedence.
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
