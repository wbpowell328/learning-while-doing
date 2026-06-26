"""
Stochastic dynamics for one simulated fund path.

All annual rates are converted to per-day quantities exactly once, here,
using the trading_days_per_year field from SimConfig.  Nothing else in the
codebase should multiply or divide by 252 (or 5, or 52).
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .result import JumpEvent


def _make_rng(session_seed: int, experiment_index: int) -> np.random.Generator:
    """
    Derive an independent, reproducible RNG for (session_seed, experiment_index).

    SeedSequence.spawn gives non-overlapping sub-streams; no collision risk
    even for large experiment_index values.
    """
    ss = np.random.SeedSequence(session_seed)
    child = ss.spawn(experiment_index + 1)[experiment_index]
    return np.random.default_rng(child)


def run_path(
    config: SimConfig,
    c_star: float,
    n_days: int,
    session_seed: int,
    experiment_index: int,
) -> tuple[
    np.ndarray,  # cash_series       (post-rebalance)
    np.ndarray,  # invested_series   (post-rebalance)
    np.ndarray,  # flow_series
    np.ndarray,  # shortfall_series  (pre-rebalance; used for cost)
    list[JumpEvent],
]:
    """
    Simulate one fund path of length n_days.

    Returns daily end-of-day arrays and the institutional jump log.
    """
    rng = _make_rng(session_seed, experiment_index)
    cfg = config
    tdy = cfg.trading_days_per_year  # shorthand; used only for conversion

    # --- Per-day parameters (converted once from annual) ---
    dt = 1.0 / tdy

    mu_daily = cfg.mu_net_annual * dt
    sigma_daily = cfg.sigma_net_annual * np.sqrt(dt)

    # Poisson rate: expected jumps per day
    jump_rate_daily = cfg.jump_rate_annual * dt

    # Invested assets grow at r_market each day
    r_market_daily = cfg.r_market_annual * dt

    # Regime drift adjustments (annual → daily)
    mu_regime_daily = tuple(m * dt for m in cfg.mu_regime_annual)

    # --- Initial state ---
    aum = cfg.initial_aum
    cash = c_star * aum
    invested = aum - cash
    regime = cfg.initial_regime

    cash_series = np.empty(n_days)
    invested_series = np.empty(n_days)
    flow_series = np.empty(n_days)
    shortfall_series = np.zeros(n_days)  # pre-rebalance shortfall per day
    event_log: list[JumpEvent] = []

    for day in range(n_days):
        # 1. Regime transition (HMM)
        if not cfg.stationary:
            if rng.random() < cfg.regime_switch_prob:
                regime = 1 - regime

        # 2. Effective daily drift (base + regime adjustment)
        mu_eff = mu_daily + mu_regime_daily[regime]

        # 3. Retail net flow: GBM increment on fraction-of-AUM basis
        retail_flow = aum * (mu_eff + sigma_daily * rng.standard_normal())

        # 4. Institutional jumps: Poisson count then lognormal sizes
        n_jumps = rng.poisson(jump_rate_daily)
        inst_flow = 0.0
        for _ in range(n_jumps):
            log_size = cfg.jump_mean_log + cfg.jump_std_log * rng.standard_normal()
            size_frac = np.exp(log_size)          # always positive
            direction = 1 if rng.random() < cfg.jump_inflow_prob else -1
            inst_flow += direction * size_frac * aum
            event_log.append(JumpEvent(day=day, size_fraction=size_frac, direction=direction))

        # 5. Total net flow hits cash first
        net_flow = retail_flow + inst_flow
        cash += net_flow

        # 6. Record shortfall BEFORE rebalancing.  The rebalancer will draw from
        #    invested assets to cover the deficit, but that forced liquidation is
        #    the economic cost we want to capture (see DECISIONS.md D8).
        shortfall_series[day] = max(-cash, 0.0)

        # 7. Invested assets earn r_market (before rebalance)
        invested *= (1.0 + r_market_daily)
        aum = cash + invested

        # 8. End-of-day rebalance: steer cash toward c_star * AUM
        target_cash = c_star * aum
        gap = target_cash - cash                    # positive → need more cash
        transfer = cfg.rebalance_speed * gap        # partial if speed < 1
        # Can only transfer what's available in invested (no leverage)
        transfer = max(transfer, -invested)         # can't take more than invested
        cash += transfer
        invested -= transfer

        # 9. Record end-of-day state
        cash_series[day] = cash
        invested_series[day] = invested
        flow_series[day] = net_flow

    return cash_series, invested_series, flow_series, shortfall_series, event_log
