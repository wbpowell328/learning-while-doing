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
    kg_vs_batch_size, kg_indep_scalar_vs_batch_size, kg_indep_beliefs_vs_batch_size,
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
    SetMStarRequest, SetMStarResponse,
    SetLengthScaleRequest, SetLengthScaleResponse,
    ExperimentRequest, ExperimentResponse,
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


def _to_display(session: "Session", value):
    """
    Convert a belief-frame quantity (posterior mean, true F(θ), etc.) to the
    frame the UI wants. Belief always minimises internally, so for a
    maximisation app the display value is −(belief-frame value).
    """
    arr = np.asarray(value, dtype=float)
    if not session.minimize:
        arr = -arr
    return arr

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
        # Reward-frame quantities (app is a MAXIMISATION problem).
        total_reward=float(getattr(result, "total_reward", -result.total_cost)),
        market_gain=float(getattr(result, "market_gain", 0.0)),
        cash_gain=float(getattr(result, "cash_gain", 0.0)),
        # `shortfall_penalty` — for cash_balance_2d this is a computed property
        # summing ind + inst; both apps expose the attribute.
        shortfall_penalty=float(getattr(result, "shortfall_penalty", 0.0)),
        # Legacy cost-frame quantities — kept so callers that still read them
        # (e.g. old tests, batch metrics) do not break.
        total_cost=float(result.total_cost),
        opportunity_cost=float(result.opportunity_cost),
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
    # `budget` retained in the signature for backward-compat with the
    # batch endpoint; the online-KG variants no longer use it (m* took
    # its place per the Warren-2026 formulation).
    if name == "ie":
        return IEPolicy(acq_cfg)
    if name == "kg":
        return KGPolicy(acq_cfg)                # offline correlated (analytic)
    if name == "kg_indep":
        return KGIndependentPolicy(acq_cfg)     # offline independent
    if name == "okg":
        return OKGCorrelatedPolicy(acq_cfg)     # online correlated (μ + KG(m*))
    if name == "okg_indep":
        return OKGIndependentPolicy(acq_cfg)    # online independent
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
        minimize=bool(app_mod.MINIMIZE),
        m_star=max(1, int(req.m_star or 1)),
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
        m_star=session.m_star,
    )


@app.post("/sessions/{sid}/m_star")
def set_m_star(sid: str, body: SetMStarRequest) -> SetMStarResponse:
    """
    Update the KG batch-size-trick multiplier m* for an existing session.
    Effective on the NEXT policy.propose() call; past steps are not
    recomputed. No-op if the session's policy isn't a KG variant.
    """
    session = _get_or_404(sid)
    session.set_m_star(int(body.m_star))
    return SetMStarResponse(m_star=session.m_star)


@app.post("/sessions/{sid}/length_scale")
def set_length_scale(sid: str, body: SetLengthScaleRequest) -> SetLengthScaleResponse:
    """
    Rebuild the belief with a new GP kernel length_scale, replaying the
    session's history through the fresh posterior. Same observations,
    different smoothness assumption — the GP posterior line/surface will
    look tighter (small length_scale) or smoother (large).

    For 1-D sessions, `length_scale` must be a scalar. For 2-D sessions,
    either a scalar (isotropic — broadcast to both dims) or a length-2
    list (ARD — per-dim scales).
    """
    session = _get_or_404(sid)
    ls = body.length_scale
    # For multi-D apps, allow a scalar and broadcast to length dim.
    if session.dim > 1 and not isinstance(ls, list):
        ls = [float(ls)] * session.dim
    if isinstance(ls, list) and len(ls) != session.dim:
        raise HTTPException(status_code=400,
            detail=f"length_scale must be scalar or length-{session.dim} list; got length {len(ls)}")
    if isinstance(ls, list):
        for v in ls:
            if not (v > 0 and v < 1e6):
                raise HTTPException(status_code=400,
                    detail=f"length_scale entries must be positive and finite; got {ls}")
    else:
        if not (ls > 0 and ls < 1e6):
            raise HTTPException(status_code=400,
                detail=f"length_scale must be positive and finite; got {ls}")
    session.set_length_scale(ls)
    return SetLengthScaleResponse(length_scale=session.belief.config.length_scale)


@app.post("/sessions/{sid}/experiment")
def experiment(sid: str, body: ExperimentRequest) -> ExperimentResponse:
    """
    "Batch" experiment run driven from the top-of-page control bar.

    1. Reset the session (empty belief, empty history, re-seeded RNG).
    2. Apply the requested policy + parameter (m* for KG family,
       z_alpha for IE, nothing for random / human).
    3. Iteration 1 = sample at `theta_init` (user-picked) for `n_days`.
    4. If K > 0: iterations 2..K+1 use the policy to pick θ, each for
       `n_days`. (For Human policy the caller should always send K=0;
       enforcing here would silently swallow input, so we let the
       caller send whatever and just execute.)

    Returns a summary; the frontend re-fetches posterior, KG chart,
    KG(m) (if advanced), and enriched history separately.
    """
    session = _get_or_404(sid)
    # --- 1. Reset ------------------------------------------------------
    session.reset()

    # --- 2. Apply policy + parameter -----------------------------------
    # Optionally adjust m* on the session before constructing the policy
    # so the new policy inherits it via set_policy.
    if body.m_star is not None:
        session.set_m_star(int(body.m_star))
    new_policy = _make_policy(body.policy, session._acq_config, budget=None)
    session.set_policy(new_policy)
    # z_alpha lives on the AcquisitionConfig. For IE, mutate it in place
    # so the IE policy picks it up on its next propose(). AcquisitionConfig
    # is a plain dataclass (not frozen).
    if body.z_alpha is not None and body.policy == "ie":
        from dataclasses import replace as _replace
        session._acq_config = _replace(session._acq_config, z_alpha=float(body.z_alpha))
        # Rebuild the policy so it captures the new config.
        session.set_policy(_make_policy(body.policy, session._acq_config, budget=None))

    n_days = int(body.n_days)
    if n_days < 1:
        raise HTTPException(status_code=400, detail=f"n_days must be >= 1; got {n_days}")

    # --- 3. Iteration 1 — user-picked θ --------------------------------
    session.evaluate(body.theta_init, n_days=n_days)

    # --- 4. Iterations 2..K+1 — policy-picked θ ------------------------
    K = max(0, int(body.K))
    for _ in range(K):
        session.step(n_days=n_days)

    # --- Response ------------------------------------------------------
    dim = session.dim
    best = session.best_impparam()
    hist_out = [
        (_theta_out(t, dim), float(v)) for t, v in session.history
    ]
    return ExperimentResponse(
        n_steps=int(session.n_steps),
        best_impparam=_theta_out(best, dim),
        history=hist_out,
    )


@app.get("/sessions/{sid}/observations_enriched")
def observations_enriched(sid: str) -> dict:
    """
    Per-observation rows with three extra values per row: the observed
    θ and reward the user already sees, plus μ^n and offline analytic
    correlated KG evaluated AT the belief state that existed BEFORE
    that observation was made — i.e. what the policy saw when it
    decided to pick that θ. Row 0 (first observation) sees the pure
    prior everywhere.

    KG is computed under the session's current m* (batch-size trick
    multiplier) so it reflects what the policy actually used.

    Values are in the app's DISPLAY frame (rewards for maximise apps,
    costs for minimise apps).
    """
    from policy.belief import BeliefModel as _BeliefModel
    session = _get_or_404(sid)
    if not session.history:
        return {"rows": []}

    cfg = session._acq_config
    # Search grid for the KG evaluation — match /kg endpoint conventions.
    if session.dim == 1:
        search_grid = np.linspace(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)
    else:
        lo = _as_list(cfg.impparam_min)
        hi = _as_list(cfg.impparam_max)
        grid_size_2d = min(int(cfg.grid_size), 25)
        axis1 = np.linspace(lo[0], hi[0], grid_size_2d)
        axis2 = np.linspace(lo[1], hi[1], grid_size_2d)
        G1, G2 = np.meshgrid(axis1, axis2, indexing="ij")
        search_grid = np.stack([G1.ravel(), G2.ravel()], axis=-1)

    # Rebuild the belief incrementally. For each row, record μ and KG at
    # its θ BEFORE that row's observation is folded in.
    fresh = _BeliefModel(session.belief.config, dim=session.dim)
    m_star = session.m_star

    rows: list[dict] = []
    for i, (impparam, disp_val) in enumerate(session.history):
        # Normalise θ to (1, dim) for posterior/KG queries.
        theta_flat = np.atleast_1d(np.asarray(impparam, dtype=float)).reshape(-1)
        theta_query = theta_flat.reshape(1, -1)

        mu_arr, _ = fresh.posterior(theta_query)
        mu_display = float(_to_display(session, mu_arr)[0])
        kg_val = float(
            kg_analytic_correlated_at(fresh, search_grid, theta_query, m_star=m_star)[0]
        )

        rows.append({
            "step": i,
            "theta": _theta_out(impparam, session.dim),
            "value": float(disp_val),
            "mu": mu_display,
            "kg": kg_val,
        })
        # Now fold this observation in for the next row's "prior" state.
        internal = float(disp_val) if session.minimize else -float(disp_val)
        fresh.update(impparam, internal)

    return {"rows": rows}


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
    mean_display = _to_display(session, mean)  # negated for maximise apps

    # History flattened as rows [theta1, theta2, value_in_display_frame]
    # (session.history already stores display-frame values.)
    hist_rows: list[list[float]] = []
    for t, c in session.history:
        arr = np.atleast_1d(np.asarray(t, dtype=float))
        hist_rows.append([float(arr[0]), float(arr[1]), float(c)])

    best = _as_list(session.best_impparam())
    return Posterior2DResponse(
        axis1=axis1.tolist(),
        axis2=axis2.tolist(),
        mean=mean_display.tolist(),
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
    mean_rewards: list[float] = []
    mean_costs: list[float] = []   # legacy — kept for backward compat
    for c in grid:
        results = [
            simulate(
                config=cfg,
                impparam=float(c),
                horizon_weeks=sc.horizon_weeks,
                session_seed=base_seed,
                experiment_index=i,
            )
            for i in range(n_reps)
        ]
        mean_rewards.append(float(np.mean([r.total_reward for r in results])))
        mean_costs.append(float(np.mean([r.total_cost   for r in results])))

    true_best_idx = int(np.argmax(mean_rewards))   # max reward = min cost
    player_c = session.best_impparam()
    player_idx = int(np.argmin(np.abs(grid - player_c)))

    naive_c = float(np.clip(0.10, cfg.impparam_min, cfg.impparam_max))
    naive_results = [
        simulate(
            config=cfg,
            impparam=naive_c,
            horizon_weeks=sc.horizon_weeks,
            session_seed=base_seed,
            experiment_index=i,
        )
        for i in range(n_reps)
    ]

    return RevealResponse(
        impparams=grid.tolist(),
        mean_reward=mean_rewards,
        mean_cost=mean_costs,
        true_best_impparam=float(grid[true_best_idx]),
        true_max_reward=mean_rewards[true_best_idx],
        true_min_cost=mean_costs[true_best_idx],
        player_best_impparam=player_c,
        player_best_reward=mean_rewards[player_idx],
        player_best_cost=mean_costs[player_idx],
        naive_reward=float(np.mean([r.total_reward for r in naive_results])),
        naive_cost=float(np.mean([r.total_cost   for r in naive_results])),
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

    # Online KG. Internally (min-cost form) Ryzhov gives
    #     online_KG(x) = μ_n(x) − (N − n) · offline_KG(x)
    # and picks argmin. For a maximise app we display
    #     display_online_KG(x) = μ_reward(x) + (N − n) · offline_KG(x)
    #                          = −μ_cost(x) + (N − n) · offline_KG(x)
    # which the user maximises — consistent with the offline KG chart above.
    # Offline KG(x) is always ≥ 0 (info value) and is unchanged by direction.
    steps_used = int(session.n_steps)
    remaining = max(0, int(budget) - steps_used)
    online_ana_internal = mu_probes - remaining * ana
    online_ind_internal = mu_probes - remaining * ind
    online_ana = _to_display(session, online_ana_internal).tolist()
    online_ind = _to_display(session, online_ind_internal).tolist()
    mu_display = _to_display(session, mu_probes).tolist()

    return KGComparisonResponse(
        impparams=probes.tolist(),
        posterior_mean=mu_display,
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


@app.get("/sessions/{sid}/kg_vs_m")
def kg_vs_m(sid: str, theta: float | None = None, theta2: float | None = None,
            m_max: int = 50, sigma_eps: float | None = None) -> dict:
    """
    KG(θ) at a single θ as a function of the effective batch size m.

    Supports 1-D and 2-D apps. For 2-D, pass both `theta` (θ₁) and
    `theta2` (θ₂); if either is omitted the endpoint uses argmax of
    the offline-correlated KG surface as the default candidate.

    Motivates the "S-curve" effect: a single experiment (m=1) at a θ with
    low signal-to-noise can produce a tiny KG, which is what makes the
    online-KG policies default to μ_reward and never explore. Batching m
    repeat observations there (noise → σ/√m) can unlock a much larger
    information gain.

    Returns { theta (scalar for 1-D or list for 2-D), m_values, kg_values,
             kg_values_correlated, kg_values_independent, noise_std,
             noise_std_belief, base_kg, base_kg_indep, delta_corr,
             sigma_tilde_corr, delta_indep, sigma_tilde_indep }.
    """
    session = _get_or_404(sid)
    if session.dim not in (1, 2):
        raise HTTPException(status_code=400,
            detail=f"kg_vs_m supports 1-D and 2-D sessions; this one has dim={session.dim}")

    cfg = session._acq_config

    if session.dim == 1:
        search_grid = np.linspace(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)
        if theta is None:
            kg_curve = kg_analytic_correlated_at(session.belief, search_grid, search_grid)
            theta_used = float(search_grid[int(np.argmax(kg_curve))])
        else:
            theta_used = float(np.clip(theta, cfg.impparam_min, cfg.impparam_max))
    else:  # dim == 2
        lo = _as_list(cfg.impparam_min)
        hi = _as_list(cfg.impparam_max)
        # Same 2-D grid convention as the /kg_2d endpoint (capped for speed).
        grid_size_2d = min(int(cfg.grid_size), 25)
        axis1 = np.linspace(lo[0], hi[0], grid_size_2d)
        axis2 = np.linspace(lo[1], hi[1], grid_size_2d)
        G1, G2 = np.meshgrid(axis1, axis2, indexing="ij")
        search_grid = np.stack([G1.ravel(), G2.ravel()], axis=-1)  # (n, 2)
        if theta is None or theta2 is None:
            kg_curve = kg_analytic_correlated_at(session.belief, search_grid, search_grid)
            theta_used = search_grid[int(np.argmax(kg_curve))].copy()
        else:
            theta_used = np.array([
                float(np.clip(theta,  lo[0], hi[0])),
                float(np.clip(theta2, lo[1], hi[1])),
            ])

    # σ_ε override is FUTURE-NOISE ONLY (the classical Frazier & Powell
    # setup that produces the S-curve): posterior stays fit under the
    # session's actual noise, override applies only to the future batch
    # measurement noise in the KG formula. Under a full refit both Δ
    # and σ̃ shrink together and the S never appears — verified with
    # Warren after multiple iterations.
    future_noise = float(sigma_eps) if sigma_eps is not None else None
    belief_for_kg = session.belief   # never refit; belief is source of truth

    m_max_int = int(max(1, min(m_max, 5000)))
    m_positive = list(range(1, m_max_int + 1))
    kg_positive_corr = kg_vs_batch_size(
        belief_for_kg, search_grid, theta_used, m_positive,
        noise_std_override=future_noise,
    )

    # Truly-independent-beliefs KG at θ: each grid point is treated as a
    # separate alternative with its own conjugate Normal-Normal belief,
    # updated only by observations that fell in its own bin. Historical
    # posterior uses the belief's actual noise; future_noise_std applies
    # only to hypothetical batch experiments in the KG expression.
    # Session history is in display frame; negate for maximise apps so
    # the belief sees "value to minimise".
    history_internal = [
        (t, (v if session.minimize else -v)) for t, v in session.history
    ]
    kg_positive_indep = kg_indep_beliefs_vs_batch_size(
        belief_for_kg, search_grid, theta_used, m_positive,
        history=history_internal,
        future_noise_std=future_noise,
    )

    # ── Diagnostic: |Δ| and σ̃(m=1) for both formulations ──────────────
    # These are what determine whether the S-curve is visible: KG has a
    # sharp transition when |Δ| is comparable to σ̃; when |Δ| ≪ σ̃ the KG
    # is near its max already at m=1 (no S), and when |Δ| ≫ σ̃ the KG
    # stays near 0 for many m (S visible late or off-plot).
    # theta_used may be a scalar (1-D) or 2-vector (2-D); normalise for
    # posterior queries and nearest-cell lookups.
    theta_vec = np.atleast_1d(np.asarray(theta_used, dtype=float)).reshape(-1)
    theta_arr = theta_vec.reshape(1, -1)  # (1, dim)
    grid_arr = search_grid.reshape(-1, theta_vec.size) if search_grid.ndim == 1 else search_grid

    mu_theta_arr, std_theta_arr = belief_for_kg.posterior(theta_arr)
    mu_theta_corr = float(mu_theta_arr[0])
    var_theta_corr = float(std_theta_arr[0]) ** 2
    mu_grid_corr, _ = belief_for_kg.posterior(grid_arr)
    mu_best_corr = float(np.min(mu_grid_corr))
    delta_corr = abs(mu_theta_corr - mu_best_corr)
    # σ_ε for future observations: user override if set, else belief default.
    session_noise = float(belief_for_kg.config.noise_std)
    future_noise_val = float(future_noise) if future_noise is not None else session_noise
    sigma_tilde_corr_1 = float(np.sqrt(var_theta_corr + future_noise_val ** 2))

    # Same quantities under the truly-independent-beliefs formulation.
    # Posterior update uses SESSION noise (the belief's actual assumption),
    # not the override — that's the whole point of the future-noise-only
    # semantics: the posterior stays "as the session learned it" while the
    # KG formula asks "what if a future batch had noise σ_ε_future?".
    # Nearest-cell bins use Euclidean distance so this works in any dim.
    prior_var = float(belief_for_kg.config.signal_std) ** 2
    prior_mean = float(belief_for_kg.config.prior_mean)
    n_grid_cells = grid_arr.shape[0]
    n_obs = np.zeros(n_grid_cells, dtype=float)
    sum_obs = np.zeros(n_grid_cells, dtype=float)
    for t_hist, v_hist in history_internal:
        t_vec = np.atleast_1d(np.asarray(t_hist, dtype=float)).reshape(-1)
        if t_vec.size != theta_vec.size:
            continue
        dists = np.linalg.norm(grid_arr - t_vec, axis=1)
        j = int(np.argmin(dists))
        n_obs[j] += 1.0
        sum_obs[j] += float(v_hist)
    prec_j = 1.0 / prior_var + n_obs / (session_noise ** 2)
    V_j_indep = 1.0 / prec_j
    mu_j_indep = V_j_indep * (prior_mean / prior_var + sum_obs / (session_noise ** 2))
    dists_theta = np.linalg.norm(grid_arr - theta_vec, axis=1)
    j_star = int(np.argmin(dists_theta))
    mu_theta_indep = float(mu_j_indep[j_star])
    V_theta_indep = float(V_j_indep[j_star])
    mask = np.ones(n_grid_cells, dtype=bool); mask[j_star] = False
    mu_best_indep = float(np.min(mu_j_indep[mask])) if mask.any() else mu_theta_indep
    delta_indep = abs(mu_theta_indep - mu_best_indep)
    # σ̃(m=1) uses the FUTURE noise (the batch measurement noise assumption).
    sigma_tilde_indep_1 = float(np.sqrt(V_theta_indep ** 2 / (V_theta_indep + future_noise_val ** 2)))
    # Prepend m=0 anchor: with zero observations no information is gained,
    # so KG(x; 0) = 0 by definition. Including it makes the "first
    # observation" jump visible, matching the classical KG(x; N) plots.
    m_values = [0] + m_positive
    kg_values = [0.0] + [float(v) for v in kg_positive_corr.tolist()]
    kg_values_indep = [0.0] + [float(v) for v in kg_positive_indep.tolist()]
    # theta_used comes back as scalar for 1-D or ndarray for 2-D; normalise
    # to a JSON-friendly type (float or list of floats).
    theta_out = float(theta_used) if session.dim == 1 else [float(v) for v in np.atleast_1d(theta_used).reshape(-1)]
    return {
        "theta": theta_out,
        "m_values": m_values,
        # Legacy field name = correlated curve (what the policy uses).
        "kg_values": kg_values,
        "kg_values_correlated": kg_values,
        "kg_values_independent": kg_values_indep,
        # Future batch noise in effect for this response (the override if
        # supplied, else the belief's own noise_std). Note this is FUTURE-
        # NOISE ONLY — the historical posterior is fit under the belief's
        # own noise_std regardless of what the caller sends.
        "noise_std": float(future_noise) if future_noise is not None
                     else float(session.belief.config.noise_std),
        "noise_std_belief": float(session.belief.config.noise_std),
        "base_kg": float(kg_positive_corr[0]) if len(kg_positive_corr) else 0.0,
        "base_kg_indep": float(kg_positive_indep[0]) if len(kg_positive_indep) else 0.0,
        # Diagnostic readouts of |Δ| and σ̃(m=1) for both formulations —
        # what governs whether the S-curve is visible or not.
        "delta_corr":        delta_corr,
        "sigma_tilde_corr":  sigma_tilde_corr_1,
        "delta_indep":       delta_indep,
        "sigma_tilde_indep": sigma_tilde_indep_1,
    }


@app.get("/sessions/{sid}/posterior")
def posterior(sid: str, grid_size: int = 200) -> PosteriorResponse:
    session = _get_or_404(sid)
    cfg = session._acq_config
    grid = np.linspace(cfg.impparam_min, cfg.impparam_max, grid_size)
    mean, std = session.belief.posterior(grid)
    return PosteriorResponse(
        impparams=grid.tolist(),
        mean=_to_display(session, mean).tolist(),  # reward frame for maximise apps
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
