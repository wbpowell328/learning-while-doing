"""
Reward accounting for one simulated path.

The fund's total reward over the horizon is
    reward = market_gain + cash_gain - shortfall_penalty
where
    market_gain      = r_market_daily * Σ invested_t
    cash_gain        = r_cash_daily   * Σ cash_t
    shortfall_penalty = r_borrow * Σ shortfall_t   (proportional, per D8;
                         no time-unit scaling)

The old cost-frame quantities (opportunity_cost = drag × cash, and the same
shortfall_cost) are still returned as an internal helper so the belief layer
can continue to minimise. Note the identity used elsewhere:

    reward = r_market * Σ AUM - opportunity_cost - shortfall_penalty

so argmax(reward) matches argmin(old cost) up to a θ-independent shift.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


def compute_rewards(
    config: SimConfig,
    impparam: float,
    cash_series: np.ndarray,        # end-of-day cash (post-rebalance)
    invested_series: np.ndarray,
    shortfall_series: np.ndarray,   # pre-rebalance shortfall per day
) -> tuple[float, float, float]:
    """
    Return (market_gain, cash_gain, shortfall_penalty) over the horizon.

    market_gain and cash_gain are non-negative by construction (rates are
    non-negative and holdings are non-negative). shortfall_penalty is
    non-negative. total_reward = market_gain + cash_gain - shortfall_penalty
    is typically positive but can be negative on a bad path.
    """
    cfg = config
    dt = 1.0 / cfg.trading_days_per_year

    r_market_daily = cfg.r_market_annual * dt
    r_cash_daily   = cfg.r_cash_annual   * dt

    market_gain = float(r_market_daily * np.sum(np.maximum(invested_series, 0.0)))
    cash_gain   = float(r_cash_daily   * np.sum(np.maximum(cash_series,     0.0)))

    # D8: shortfall penalty is a proportional friction cost, not rate × time.
    # r_borrow is the cost per dollar of forced liquidation.
    shortfall_penalty = float(cfg.r_borrow_annual * np.sum(shortfall_series))

    return market_gain, cash_gain, shortfall_penalty


def compute_costs(
    config: SimConfig,
    impparam: float,
    cash_series: np.ndarray,
    invested_series: np.ndarray,
    shortfall_series: np.ndarray,
) -> tuple[float, float]:
    """
    Legacy cost-frame view: (opportunity_cost, shortfall_cost).

    Kept so the internals of the belief/acquisition layer can keep operating
    on a "value to minimise" (= -reward, up to a θ-nearly-constant shift).
    """
    cfg = config
    dt = 1.0 / cfg.trading_days_per_year
    drag_rate_daily = (cfg.r_market_annual - cfg.r_cash_annual) * dt

    aum_series = cash_series + invested_series
    if cfg.opp_cost_on_total_cash:
        opp_basis = cash_series
    else:
        target_cash = impparam * aum_series
        opp_basis = np.maximum(cash_series - target_cash, 0.0)

    opportunity_cost = float(np.sum(drag_rate_daily * np.maximum(opp_basis, 0.0)))
    shortfall_cost = float(cfg.r_borrow_annual * np.sum(shortfall_series))

    return opportunity_cost, shortfall_cost
