import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sim import SimConfig, simulate
from policy import (
    BeliefConfig, AcquisitionConfig, SessionConfig,
    RandomPolicy, IEPolicy, KGPolicy, Session,
    KGMCPolicy, KGIndependentPolicy, OKGCorrelatedPolicy, OKGIndependentPolicy,
    kg_analytic_correlated_at, kg_mc_correlated_at, kg_independent_at,
)
from .models import (
    CreateSessionRequest, CreateSessionResponse,
    StepResponse, JumpEventOut,
    EvaluateRequest,
    ObserveRequest, ObserveResponse,
    StateResponse, PosteriorResponse,
    RevealResponse, KGComparisonResponse,
    BatchRequest, BatchResponse, BatchPolicyResult,
)

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
    return StepResponse(
        c_star=result.c_star,
        total_cost=result.total_cost,
        opportunity_cost=result.opportunity_cost,
        shortfall_cost=result.shortfall_cost,
        days=result.days,
        n_steps=session.n_steps,
        best_c_star=session.best_c_star(),
        cash_series=result.cash_series.tolist(),
        event_log=[
            JumpEventOut(day=e.day, size_fraction=e.size_fraction, direction=e.direction)
            for e in result.event_log
        ],
        initial_aum=session._sim_config.initial_aum,
    )


def _make_policy(name: str, acq_cfg: AcquisitionConfig):
    if name == "ie":
        return IEPolicy(acq_cfg)
    if name == "kg":
        return KGPolicy(acq_cfg)
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
    sim_cfg    = SimConfig(**req.sim_config.model_dump())
    belief_cfg = BeliefConfig(**req.belief_config.model_dump())
    acq_cfg    = AcquisitionConfig(**req.acq_config.model_dump())
    ses_cfg    = SessionConfig(**req.session_config.model_dump())
    policy     = _make_policy(req.policy, acq_cfg)

    session = Session(sim_cfg, belief_cfg, acq_cfg, ses_cfg, policy, req.session_seed)
    sid = str(uuid4())
    _sessions[sid] = session

    return CreateSessionResponse(session_id=sid, policy=req.policy)


@app.post("/sessions/{sid}/step")
def step(sid: str) -> StepResponse:
    session = _get_or_404(sid)
    result = session.step()
    return _make_step_response(result, session)


@app.post("/sessions/{sid}/observe")
def observe(sid: str, body: ObserveRequest) -> ObserveResponse:
    session = _get_or_404(sid)
    session.observe(body.c_star, body.total_cost)
    return ObserveResponse(
        n_observations=session.belief.n_observations,
        best_c_star=session.best_c_star(),
    )


@app.get("/sessions/{sid}/state")
def state(sid: str) -> StateResponse:
    session = _get_or_404(sid)
    return StateResponse(
        n_steps=session.n_steps,
        n_observations=session.belief.n_observations,
        best_c_star=session.best_c_star(),
        history=session.history,
    )


@app.post("/sessions/{sid}/evaluate")
def evaluate(sid: str, body: EvaluateRequest) -> StepResponse:
    session = _get_or_404(sid)
    result = session.evaluate(body.c_star)
    return _make_step_response(result, session)


@app.get("/sessions/{sid}/reveal")
def reveal(sid: str, grid_size: int = 30, n_reps: int = 12) -> RevealResponse:
    session = _get_or_404(sid)
    cfg = session._sim_config
    sc = session._sc
    base_seed = session._session_seed + 999_000

    grid = np.linspace(cfg.c_star_min, cfg.c_star_max, grid_size)
    mean_costs: list[float] = []
    for c in grid:
        costs = [
            simulate(
                config=cfg,
                c_star=float(c),
                horizon_weeks=sc.horizon_weeks,
                session_seed=base_seed,
                experiment_index=i,
            ).total_cost
            for i in range(n_reps)
        ]
        mean_costs.append(float(np.mean(costs)))

    true_best_idx = int(np.argmin(mean_costs))
    player_c = session.best_c_star()
    player_idx = int(np.argmin(np.abs(grid - player_c)))

    naive_c = float(np.clip(0.10, cfg.c_star_min, cfg.c_star_max))
    naive_costs = [
        simulate(
            config=cfg,
            c_star=naive_c,
            horizon_weeks=sc.horizon_weeks,
            session_seed=base_seed,
            experiment_index=i,
        ).total_cost
        for i in range(n_reps)
    ]

    return RevealResponse(
        c_stars=grid.tolist(),
        mean_cost=mean_costs,
        true_best_c_star=float(grid[true_best_idx]),
        true_min_cost=mean_costs[true_best_idx],
        player_best_c_star=player_c,
        player_best_cost=mean_costs[player_idx],
        naive_cost=float(np.mean(naive_costs)),
    )


@app.post("/sessions/batch")
def batch_run(req: BatchRequest) -> BatchResponse:
    """
    Run an entire policy family, aggregating results across sims_per_policy
    independent runs (each with a distinct session seed but shared across all
    policies within a sim_idx — Common Random Numbers reduces variance).

    For each run: create session, take `budget` steps, record final best_c_star
    and cumulative simulated cost. After the loop, evaluate the final best_c_star
    at ground truth (12-rep MC).
    """
    sim_cfg    = SimConfig(**req.sim_config.model_dump())
    belief_cfg = BeliefConfig(**req.belief_config.model_dump())
    acq_cfg    = AcquisitionConfig(**req.acq_config.model_dump())
    ses_cfg    = SessionConfig(**req.session_config.model_dump())

    # ------------------------------------------------------------------
    # Build (label, param, factory) for each policy in the family
    # ------------------------------------------------------------------
    if req.family == "KG":
        family = [
            ("KG offline correlated (analytic)", 0.0, lambda: KGPolicy(acq_cfg)),
            ("KG offline correlated (MC, n=500)", 1.0, lambda: KGMCPolicy(acq_cfg, n_mc=500)),
            ("KG offline independent", 2.0, lambda: KGIndependentPolicy(acq_cfg)),
            ("KG online correlated (Ryzhov)", 3.0, lambda: OKGCorrelatedPolicy(acq_cfg, budget=req.budget)),
            ("KG online independent", 4.0, lambda: OKGIndependentPolicy(acq_cfg, budget=req.budget)),
        ]
    elif req.family == "IE":
        # z_alpha values: 0, 0.2, 0.4, ..., 4.0 → 21 policies
        family = []
        for k in range(21):
            z = round(k * 0.2, 3)
            # Rebuild AcqConfig with this z_alpha
            cfg_z = AcquisitionConfig(
                c_star_min=acq_cfg.c_star_min,
                c_star_max=acq_cfg.c_star_max,
                grid_size=acq_cfg.grid_size,
                z_alpha=z,
            )
            family.append((f"IE (z={z:.2f})", z, (lambda c=cfg_z: IEPolicy(c))))
    else:
        raise HTTPException(400, f"unknown family: {req.family}")

    # ------------------------------------------------------------------
    # Run each policy sims_per_policy times
    # ------------------------------------------------------------------
    per_policy: dict[str, dict] = {label: {"param": param, "best": [], "term": [], "cum": []}
                                    for label, param, _ in family}

    for sim_idx in range(req.sims_per_policy):
        base_seed = req.session_seed + sim_idx * 10_000
        for label, _, factory in family:
            session = Session(sim_cfg, belief_cfg, acq_cfg, ses_cfg, factory(), base_seed)
            cumulative = 0.0
            for _ in range(req.budget):
                result = session.step()
                cumulative += result.total_cost
            best = session.best_c_star()
            term = _evaluate_expected_cost(sim_cfg, ses_cfg, best, base_seed + 999_000, n_reps=12)
            per_policy[label]["best"].append(best)
            per_policy[label]["term"].append(term)
            per_policy[label]["cum"].append(cumulative)

    # ------------------------------------------------------------------
    # Ground truth: best_c_star and min_cost over a fine grid
    # ------------------------------------------------------------------
    true_best, true_min = _ground_truth(sim_cfg, ses_cfg, acq_cfg,
                                         base_seed_prefix=req.session_seed + 777_000,
                                         grid_size=30, n_reps=12)

    # Aggregate + assemble response
    def _agg(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std())

    results: list[BatchPolicyResult] = []
    for label, param, _ in family:
        d = per_policy[label]
        m_best, s_best = _agg(d["best"])
        m_term, s_term = _agg(d["term"])
        m_cum, s_cum = _agg(d["cum"])
        results.append(BatchPolicyResult(
            policy=label, param=param,
            mean_best_c_star=m_best, std_best_c_star=s_best,
            mean_terminal_cost=m_term, std_terminal_cost=s_term,
            mean_cumulative_cost=m_cum, std_cumulative_cost=s_cum,
            best_c_stars=d["best"],
            terminal_costs=d["term"],
            cumulative_costs=d["cum"],
        ))

    return BatchResponse(
        family=req.family,
        sims_per_policy=req.sims_per_policy,
        budget=req.budget,
        session_seed=req.session_seed,
        true_best_c_star=true_best,
        true_min_cost=true_min,
        policies=results,
    )


def _evaluate_expected_cost(sim_cfg: SimConfig, ses_cfg: SessionConfig,
                             c_star: float, base_seed: int, n_reps: int = 12) -> float:
    costs = [
        simulate(config=sim_cfg, c_star=float(c_star),
                 horizon_weeks=ses_cfg.horizon_weeks,
                 session_seed=base_seed, experiment_index=i).total_cost
        for i in range(n_reps)
    ]
    return float(np.mean(costs))


def _ground_truth(sim_cfg: SimConfig, ses_cfg: SessionConfig, acq_cfg: AcquisitionConfig,
                   base_seed_prefix: int, grid_size: int = 30, n_reps: int = 12) -> tuple[float, float]:
    grid = np.linspace(acq_cfg.c_star_min, acq_cfg.c_star_max, grid_size)
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
    mc_samples: int = 500,
    mc_seed: int = 12345,
    budget: int = 10,
) -> KGComparisonResponse:
    """
    KG at a coarse probe grid (default 5% spacing across [c_star_min, c_star_max]).

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

    # Build probe grid: multiples of `spacing` inside [c_star_min, c_star_max].
    first = float(np.ceil(cfg.c_star_min / spacing) * spacing)
    if first < cfg.c_star_min - 1e-9:
        first += spacing
    probes = np.arange(first, cfg.c_star_max + spacing * 1e-6, spacing)
    if probes.size == 0:
        probes = np.array([0.5 * (cfg.c_star_min + cfg.c_star_max)])

    # Search grid: the same 100-point grid the KG policy uses.
    search_grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

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
        c_stars=probes.tolist(),
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
    grid = np.linspace(cfg.c_star_min, cfg.c_star_max, grid_size)
    mean, std = session.belief.posterior(grid)
    return PosteriorResponse(
        c_stars=grid.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        best_c_star=session.best_c_star(),
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
