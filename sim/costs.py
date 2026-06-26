"""
Cost accounting for one simulated path.

Opportunity cost uses the annual drag rate converted to per-day (÷ trading_days).
Shortfall cost is a PROPORTIONAL PENALTY (see DECISIONS.md D8): r_borrow × shortfall,
with no time-unit scaling.  r_borrow is therefore a penalty rate per dollar of
forced liquidation, not a daily interest rate.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


def compute_costs(
    config: SimConfig,
    c_star: float,
    cash_series: np.ndarray,        # end-of-day cash (post-rebalance)
    invested_series: np.ndarray,
    shortfall_series: np.ndarray,   # pre-rebalance shortfall per day
) -> tuple[float, float]:
    """
    Return (opportunity_cost, shortfall_cost) over the horizon.

    Both values are non-negative by construction.
    """
    cfg = config
    dt = 1.0 / cfg.trading_days_per_year

    drag_rate_daily = (cfg.r_market_annual - cfg.r_cash_annual) * dt

    aum_series = cash_series + invested_series

    if cfg.opp_cost_on_total_cash:
        # D1-A: drag on ALL cash held
        opp_basis = cash_series
    else:
        # D1-B: drag only on cash above target
        target_cash = c_star * aum_series
        opp_basis = np.maximum(cash_series - target_cash, 0.0)

    opportunity_cost = float(np.sum(drag_rate_daily * np.maximum(opp_basis, 0.0)))

    # D8: shortfall cost is a proportional penalty, NOT rate × time.
    # r_borrow is the friction cost per dollar of forced asset liquidation.
    shortfall_cost = float(cfg.r_borrow_annual * np.sum(shortfall_series))

    return opportunity_cost, shortfall_cost
