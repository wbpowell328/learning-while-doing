"""
Cost accounting for one simulated fund path — 2-parameter variant.

Three cost streams:

  opportunity_cost   drag on cash held (same formulation as the 1D version).
  shortfall_ind      deferred-payment penalty for individual-caused shortfalls
                     (proportional to r_borrow_ind_annual, typically small).
  shortfall_inst     forced-liquidation penalty for institutional-caused shortfalls
                     (proportional to r_borrow_inst_annual, typically large).

Neither shortfall penalty is scaled by time — both r_borrow rates are
"friction per dollar of shortfall," not daily rates.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


def compute_costs(
    config: SimConfig,
    impparam: np.ndarray,               # shape (2,): (theta_ind, theta_inst)
    cash_series: np.ndarray,
    invested_series: np.ndarray,
    aum_ind_series: np.ndarray,
    aum_inst_series: np.ndarray,
    shortfall_ind_series: np.ndarray,
    shortfall_inst_series: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return (opportunity_cost, shortfall_ind_cost, shortfall_inst_cost) over the horizon.
    All non-negative by construction.
    """
    cfg = config
    theta = np.asarray(impparam, dtype=float).reshape(-1)
    theta_ind, theta_inst = float(theta[0]), float(theta[1])
    dt = 1.0 / cfg.trading_days_per_year

    drag_rate_daily = (cfg.r_market_annual - cfg.r_cash_annual) * dt

    if cfg.opp_cost_on_total_cash:
        opp_basis = cash_series
    else:
        target_cash = theta_ind * aum_ind_series + theta_inst * aum_inst_series
        opp_basis = np.maximum(cash_series - target_cash, 0.0)

    opportunity_cost = float(np.sum(drag_rate_daily * np.maximum(opp_basis, 0.0)))

    shortfall_ind_cost  = float(cfg.r_borrow_ind_annual  * np.sum(shortfall_ind_series))
    shortfall_inst_cost = float(cfg.r_borrow_inst_annual * np.sum(shortfall_inst_series))

    return opportunity_cost, shortfall_ind_cost, shortfall_inst_cost
