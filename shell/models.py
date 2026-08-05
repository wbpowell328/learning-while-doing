"""Pydantic request / response schemas for the shell API."""
from __future__ import annotations
from typing import Any, Literal, Union
from pydantic import BaseModel, ConfigDict


# θ is a scalar for 1-D apps and a vector for multi-dim apps. Endpoints
# accept either; responses carry `dim` metadata so the frontend knows the
# shape.
ThetaLike = Union[float, list[float]]


# ---------------------------------------------------------------------------
# Request schemas — mirror the frozen dataclasses with identical defaults
# ---------------------------------------------------------------------------

class SimConfigIn(BaseModel):
    """1-D cash_balance sim config (defaults shown). Passed through untyped
    fields are dropped by pydantic — for 2-D apps, use `sim_config_dict`
    on CreateSessionRequest or pass through the untyped `extra` fields."""
    model_config = ConfigDict(extra="allow")   # accept extra fields for other apps

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
    impparam_min: Union[float, list[float]] = 0.01
    impparam_max: Union[float, list[float]] = 0.20
    trading_days_per_year: int = 252


class BeliefConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Defaults are for the REWARD frame (cash_balance is a maximise app).
    # Typical run reward ~$35k, ~$1-3k variation across θ, per-run noise
    # ~$1-2k. Session negates prior_mean before feeding to the belief for
    # maximise apps (see policy/session.py).
    length_scale: Union[float, list[float]] = 0.04
    signal_std: float = 3_000.0
    noise_std: float = 1_500.0
    prior_mean: float = 35_000.0
    jitter: float = 1e-6


class AcqConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    impparam_min: Union[float, list[float]] = 0.01
    impparam_max: Union[float, list[float]] = 0.20
    grid_size: int = 100
    z_alpha: float = 0.0


class SessionConfigIn(BaseModel):
    horizon_weeks: int = 26
    best_impparam_grid: int = 200


class CreateSessionRequest(BaseModel):
    # NEW: which application to instantiate. Uses the apps/ registry key.
    app_name: str = "cash_balance"
    sim_config: SimConfigIn = SimConfigIn()
    belief_config: BeliefConfigIn = BeliefConfigIn()
    acq_config: AcqConfigIn = AcqConfigIn()
    session_config: SessionConfigIn = SessionConfigIn()
    policy: Literal[
        "random",
        "ie",
        "kg",         # offline correlated (analytic) — default KG
        "kg_indep",   # offline independent
        "okg",        # online correlated (Ryzhov)
        "okg_indep",  # online independent
        "human",
    ] = "human"
    # Required by the online-KG policies (they need N to compute (N-n) in the
    # Ryzhov formula). Ignored by the offline policies; the frontend still
    # tracks its own budget for the human/auto-run loop.
    budget: int | None = None
    # Batch-size-trick multiplier for KG-family policies. m_star>1 evaluates
    # KG as if we were going to run m_star repeat obs at the chosen θ
    # (scales noise variance by 1/m_star), even though we only run one.
    # Default 1 (classical single-shot KG). Ignored by non-KG policies.
    m_star: int = 1
    session_seed: int = 42


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CreateSessionResponse(BaseModel):
    session_id: str
    policy: str
    app_name: str
    dim: int                  # θ dimension (1 for cash_balance, 2 for cash_balance_2d)
    minimize: bool
    impparam_min: list[float] # always returned as a list (length dim)
    impparam_max: list[float]
    m_star: int = 1           # batch-size-trick multiplier for KG-family policies


class SetMStarRequest(BaseModel):
    m_star: int


class SetMStarResponse(BaseModel):
    m_star: int


class SetLengthScaleRequest(BaseModel):
    # Scalar for 1-D or list-of-dim for 2-D (ARD).
    length_scale: Union[float, list[float]]


class SetLengthScaleResponse(BaseModel):
    length_scale: Union[float, list[float]]


class ExperimentRequest(BaseModel):
    """
    One "experiment" — a full policy run controlled from the top-of-page
    bar. Semantics: reset the session (empty belief, empty history), then
    execute K+1 iterations. Iteration 1 samples at the user-picked
    theta_init; iterations 2..K+1 have the chosen policy pick θ.
    """
    theta_init: ThetaLike        # scalar for 1-D, list[float] for 2-D
    n_days: int                  # days per iteration
    policy: str                  # 'kg' | 'kg_indep' | 'okg' | 'okg_indep' | 'ie' | 'random' | 'human'
    K: int = 0                   # number of policy-driven iterations after iter 1
    # Policy parameters — only the one that applies to `policy` is used.
    m_star: int | None = None    # KG-family batch-size trick multiplier
    z_alpha: float | None = None # IE upper-confidence-bound coefficient


class ExperimentResponse(BaseModel):
    n_steps: int
    best_impparam: ThetaLike
    # history rows: [theta, display-frame value]
    history: list[tuple[ThetaLike, float]]


class OneMoreRequest(BaseModel):
    """
    Do K+1 policy-driven iterations on the CURRENT session state
    (no reset). Symmetric with ExperimentRequest so the ExperimentBar's
    Repeat box governs both "Run" (from scratch) and "One more" (from
    the current state). K=0 means exactly one iteration (the original
    behavior). Iteration horizon is `n_days`. If `policy` is given
    (and its associated parameter), swap the session's policy before
    stepping so the user can flip policies without losing history.
    """
    n_days: int
    K: int = 0
    policy: str | None = None
    m_star: int | None = None
    z_alpha: float | None = None


class JumpEventOut(BaseModel):
    day: int
    size_fraction: float   # |jump| / AUM at jump time
    direction: int         # +1 = inflow, -1 = outflow


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="allow")   # per-app extras (e.g. cash_series)

    impparam: ThetaLike                    # scalar (1D) or list (2D+)
    # Reward-frame (app is a MAXIMISATION problem)
    total_reward: float = 0.0
    market_gain: float = 0.0
    cash_gain: float = 0.0
    shortfall_penalty: float = 0.0
    # Legacy cost-frame — retained so old clients still parse. Belief layer
    # keeps operating in cost frame internally.
    total_cost: float
    opportunity_cost: float
    shortfall_cost: float
    days: int
    n_steps: int
    best_impparam: ThetaLike
    initial_aum: float


class ObserveRequest(BaseModel):
    impparam: ThetaLike
    total_cost: float


class ObserveResponse(BaseModel):
    n_observations: int
    best_impparam: ThetaLike


class StateResponse(BaseModel):
    n_steps: int
    n_observations: int
    best_impparam: ThetaLike
    history: list[tuple[ThetaLike, float]]


class EvaluateRequest(BaseModel):
    impparam: ThetaLike


class PosteriorResponse(BaseModel):
    """1-D posterior curve. Used only for 1-D apps."""
    impparams: list[float]
    mean: list[float]
    std: list[float]
    best_impparam: float


class Posterior2DResponse(BaseModel):
    """
    2-D posterior surface. Used for 2-D apps to feed the 3-D belief plot.

    * axis1 / axis2 : the two coordinate axes (length grid_size each).
    * mean, std     : row-major flattened grids of length grid_size**2,
                      matching np.meshgrid(axis1, axis2, indexing='ij').
    * history       : list of (theta1, theta2, cost) observations.
    * best_impparam : argmin of posterior mean, as [theta1, theta2].
    """
    axis1: list[float]
    axis2: list[float]
    mean: list[float]
    std: list[float]
    history: list[list[float]]            # rows are [theta1, theta2, cost]
    best_impparam: list[float]


class KG2DResponse(BaseModel):
    """
    2-D Knowledge Gradient surface — KG(θ) at every point on a grid_size ×
    grid_size grid.  Same axis convention as Posterior2DResponse.

    * kg  : row-major flattened array of length grid_size**2, matching
            np.meshgrid(axis1, axis2, indexing='ij').  Non-negative by
            theory (small negatives are floating-point roundoff).
    * history : same observation list as Posterior2DResponse — but note that
            KG has no natural z-value at each observation (observations
            don't sit on the KG surface), so the frontend should render
            them as base-plane markers rather than surface dots.
    """
    axis1: list[float]
    axis2: list[float]
    kg: list[float]
    history: list[list[float]]
    best_impparam: list[float]


class RevealResponse(BaseModel):
    impparams: list[float]
    # Reward-frame — what the UI shows for maximise apps
    mean_reward: list[float] = []
    true_max_reward: float = 0.0
    player_best_reward: float = 0.0
    naive_reward: float = 0.0
    # Legacy cost-frame — kept for backward compat with any older reader
    mean_cost: list[float]
    true_best_impparam: float
    true_min_cost: float
    player_best_impparam: float
    player_best_cost: float
    naive_cost: float


class BatchRequest(BaseModel):
    """
    Run an entire family of policies, sims_per_policy times each, all sharing
    the same base configs. Reports aggregated outcome metrics per policy.
    """
    family: Literal["KG", "IE"]
    sims_per_policy: int = 10
    budget: int = 10                          # measurement steps per session
    sim_config: SimConfigIn = SimConfigIn()
    belief_config: BeliefConfigIn = BeliefConfigIn()
    acq_config: AcqConfigIn = AcqConfigIn()
    session_config: SessionConfigIn = SessionConfigIn()
    session_seed: int = 42


class BatchPolicyResult(BaseModel):
    policy: str                               # human-readable label
    param: float                              # numeric parameter (z_alpha for IE, arbitrary for KG)
    # Aggregates across sims (mean and std)
    mean_best_impparam: float
    std_best_impparam: float
    mean_terminal_cost: float                 # E[cost at final best_impparam], MC over noise
    std_terminal_cost: float
    mean_cumulative_cost: float               # sum of observed costs across the budget
    std_cumulative_cost: float
    # Per-sim raw arrays for plotting error bars / distributions
    best_impparams: list[float]
    terminal_costs: list[float]
    cumulative_costs: list[float]


class BatchResponse(BaseModel):
    family: str
    sims_per_policy: int
    budget: int
    session_seed: int
    # Ground-truth best θ (via MC on a fine grid), for reference
    true_best_impparam: float
    true_min_cost: float
    policies: list[BatchPolicyResult]


class KGComparisonResponse(BaseModel):
    """
    KG values at a coarse probe grid, computed three ways for pedagogical
    comparison. All three offline series share the same underlying GP posterior.

    Also returned: two "online KG" series (Ryzhov 2010, min-cost form)
        online_KG(x) = mu_n(x) - (N - n) * offline_KG(x)
    with N = budget (measurement horizon) and n = steps used so far.
    """
    impparams: list[float]                    # probe grid
    posterior_mean: list[float]             # mu_n(x) at each probe point
    # Offline KG — value-of-information only
    analytic_correlated: list[float]        # exact FPD closed form
    mc_correlated: list[float]              # Monte-Carlo estimate of same quantity
    independent: list[float]                # independent-beliefs closed form
    # Online KG — expected-cost + info-value composite (Ryzhov)
    online_correlated: list[float]          # mu - (N-n) * analytic_correlated
    online_independent: list[float]         # mu - (N-n) * independent
    # Metadata driving the online formula
    budget: int                             # N
    steps_used: int                         # n
    mc_samples: int                         # sample count that produced mc_correlated
    mc_seed: int                            # seed used for reproducibility
