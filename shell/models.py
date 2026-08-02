"""Pydantic request / response schemas for the shell API."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request schemas — mirror the frozen dataclasses with identical defaults
# ---------------------------------------------------------------------------

class SimConfigIn(BaseModel):
    initial_aum: float = 1_000_000.0
    mu_net_annual: float = 0.00
    sigma_net_annual: float = 0.02
    jump_rate_annual: float = 12.0
    jump_mean_log: float = -4.5
    jump_std_log: float = 0.5
    jump_inflow_prob: float = 0.5
    stationary: bool = False
    mu_regime_annual: tuple[float, float] = (0.001, -0.001)
    regime_switch_prob: float = 0.01
    initial_regime: int = 0
    r_market_annual: float = 0.07
    r_cash_annual: float = 0.04
    r_borrow_annual: float = 0.10
    opp_cost_on_total_cash: bool = True
    rebalance_speed: float = 1.0
    c_star_min: float = 0.01
    c_star_max: float = 0.20
    trading_days_per_year: int = 252


class BeliefConfigIn(BaseModel):
    length_scale: float = 0.04
    signal_std: float = 5_000.0
    noise_std: float = 3_000.0
    prior_mean: float = 5_000.0
    jitter: float = 1e-6


class AcqConfigIn(BaseModel):
    c_star_min: float = 0.01
    c_star_max: float = 0.20
    grid_size: int = 100


class SessionConfigIn(BaseModel):
    horizon_weeks: int = 26
    best_c_star_grid: int = 200


class CreateSessionRequest(BaseModel):
    sim_config: SimConfigIn = SimConfigIn()
    belief_config: BeliefConfigIn = BeliefConfigIn()
    acq_config: AcqConfigIn = AcqConfigIn()
    session_config: SessionConfigIn = SessionConfigIn()
    policy: Literal["random", "ie", "kg", "human"] = "human"
    session_seed: int = 42


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CreateSessionResponse(BaseModel):
    session_id: str
    policy: str


class JumpEventOut(BaseModel):
    day: int
    size_fraction: float   # |jump| / AUM at jump time
    direction: int         # +1 = inflow, -1 = outflow


class StepResponse(BaseModel):
    c_star: float
    total_cost: float
    opportunity_cost: float
    shortfall_cost: float
    days: int
    n_steps: int
    best_c_star: float
    cash_series: list[float]
    event_log: list[JumpEventOut]
    initial_aum: float


class ObserveRequest(BaseModel):
    c_star: float
    total_cost: float


class ObserveResponse(BaseModel):
    n_observations: int
    best_c_star: float


class StateResponse(BaseModel):
    n_steps: int
    n_observations: int
    best_c_star: float
    history: list[tuple[float, float]]


class EvaluateRequest(BaseModel):
    c_star: float


class PosteriorResponse(BaseModel):
    c_stars: list[float]
    mean: list[float]
    std: list[float]
    best_c_star: float


class RevealResponse(BaseModel):
    c_stars: list[float]
    mean_cost: list[float]
    true_best_c_star: float
    true_min_cost: float
    player_best_c_star: float
    player_best_cost: float
    naive_cost: float


class KGComparisonResponse(BaseModel):
    """
    KG values at a coarse probe grid, computed three ways for pedagogical
    comparison. All three share the same underlying GP posterior.
    """
    c_stars: list[float]                    # probe grid (e.g. 5% spacing)
    analytic_correlated: list[float]        # exact FPD closed form
    mc_correlated: list[float]              # Monte-Carlo estimate of same quantity
    independent: list[float]                # independent-beliefs closed form
    mc_samples: int                         # sample count that produced mc_correlated
    mc_seed: int                            # seed used for reproducibility
