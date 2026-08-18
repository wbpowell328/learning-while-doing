from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    # --- Fund ---
    initial_aum: float = 1_000_000.0

    # --- Retail flow: GBM on (net_flow / AUM) ---
    mu_net_annual: float = 0.00     # annual drift of net retail flow / AUM
    sigma_net_annual: float = 0.02  # annual vol  of net retail flow / AUM

    # --- Institutional jumps: Poisson arrivals, lognormal size ---
    jump_rate_annual: float = 12.0  # expected jumps per year
    jump_mean_log: float = -4.5     # mean of log(|jump| / AUM); median ≈ 1.1% AUM
    jump_std_log: float = 0.5       # std  of log(|jump| / AUM)
    jump_inflow_prob: float = 0.5   # P(jump is an inflow)

    # --- Profit noise (does NOT affect cash dynamics) ---
    # Independent mean-0, $/day noise added to the daily profit only, to
    # model transaction errors. Raises the observation noise the learning
    # policy sees without shifting the true optimum (mean-0). Drawn from a
    # separate RNG stream so the cash/flow path is byte-identical.
    sigma_trans: float = 750.0      # σ^trans — std of daily profit noise ($/day)

    # --- Market regime: 2-state HMM on drift ---
    stationary: bool = False                                 # True → no switching
    mu_regime_annual: tuple[float, float] = (0.001, -0.001) # per-regime drift add
    regime_switch_prob: float = 0.01                         # daily P(switch)
    initial_regime: int = 0                                  # 0=normal, 1=stress

    # --- Rates (ALL ANNUAL; converted in costs.py) ---
    r_market_annual: float = 0.07
    r_cash_annual: float = 0.04
    r_borrow_annual: float = 0.10

    # --- Opportunity cost formulation (D1 — see DECISIONS.md) ---
    # True  → drag on ALL cash held × (r_market - r_cash)
    # False → drag only on cash ABOVE θ × AUM
    opp_cost_on_total_cash: bool = True

    # --- Rebalancing ---
    # 1.0 = full end-of-day rebalance to θ × AUM; <1.0 = partial (Phase 2)
    rebalance_speed: float = 1.0

    # --- θ domain ---
    impparam_min: float = 0.01
    impparam_max: float = 0.20

    # --- Calendar ---
    trading_days_per_year: int = 252
