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
    kg_analytic_correlated_at, kg_mc_correlated_at, kg_independent_at,
)
from .models import (
    CreateSessionRequest, CreateSessionResponse,
    StepResponse, JumpEventOut,
    EvaluateRequest,
    ObserveRequest, ObserveResponse,
    StateResponse, PosteriorResponse,
    RevealResponse, KGComparisonResponse,
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


@app.get("/sessions/{sid}/kg")
def kg_comparison(
    sid: str,
    spacing: float = 0.05,
    mc_samples: int = 500,
    mc_seed: int = 12345,
) -> KGComparisonResponse:
    """
    KG at a coarse probe grid (default 5% spacing across [c_star_min, c_star_max]),
    computed three ways for side-by-side comparison:
      - analytic_correlated: exact FPD closed form (the one the policy uses)
      - mc_correlated:       Monte-Carlo estimate of the same quantity
      - independent:         closed form assuming zero cross-covariance
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

    return KGComparisonResponse(
        c_stars=probes.tolist(),
        analytic_correlated=ana.tolist(),
        mc_correlated=mc.tolist(),
        independent=ind.tolist(),
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
