"""
Stochastic dynamics for one simulated fund path — 2-parameter variant.

Two investor classes tracked separately:
  aum_ind   evolves under Brownian flows (mu_ind, sigma_ind)   — small, frequent.
  aum_inst  evolves under Poisson × lognormal jumps            — rare, large.

One shared cash pool. End-of-day rebalance target =
    theta[0] * aum_ind + theta[1] * aum_inst.

Shortfall attribution — when cash < 0 after the day's flows:
  * Institutional caused portion   = min(|cash|, |inst_outflow_today|).
    Forces immediate liquidation from `invested`; high penalty rate later.
  * Individual caused portion (rest) is deferred; low penalty rate later.

All annual rates converted to per-day quantities here (using
trading_days_per_year from SimConfig).  Nothing else in the codebase
should multiply/divide by 252.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .result import JumpEvent


def sample_flows(
    config: SimConfig,
    session_seed: int,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample-path helper for the deposits/redemptions report. Mirrors the
    exogenous portion of run_path but skips rebalancing, cost, and any
    dependence on θ. Returns per-day signed net flows:
      ind_flow_daily[t]   = individual investor net flow on day t
      inst_flow_daily[t]  = institutional investor net flow on day t

    Positive = deposit, negative = redemption. Individual AUM feeds
    back into itself; institutional AUM feeds back into itself.
    Sufficient for a diagnostic sample-path view. Uses a dedicated
    spawn_key so it doesn't collide with policy-run RNG streams.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=session_seed, spawn_key=(0xF10C5A,)),
    )
    cfg = config
    dt = 1.0 / cfg.trading_days_per_year
    mu_ind_daily    = cfg.mu_ind_annual * dt
    sigma_ind_daily = cfg.sigma_ind_annual * np.sqrt(dt)
    jump_rate_daily = cfg.jump_rate_inst_annual * dt
    mu_regime_daily = tuple(m * dt for m in cfg.mu_regime_annual)

    aum_ind  = cfg.initial_aum_ind_fraction * cfg.initial_aum
    aum_inst = cfg.initial_aum - aum_ind
    regime = cfg.initial_regime

    ind = np.empty(n_days)
    inst = np.empty(n_days)
    for day in range(n_days):
        if not cfg.stationary and rng.random() < cfg.regime_switch_prob:
            regime = 1 - regime
        mu_eff = mu_ind_daily + mu_regime_daily[regime]
        flow_ind = aum_ind * (mu_eff + sigma_ind_daily * rng.standard_normal())
        n_jumps = rng.poisson(jump_rate_daily)
        flow_inst = 0.0
        for _ in range(n_jumps):
            log_size = cfg.jump_mean_log_inst + cfg.jump_std_log_inst * rng.standard_normal()
            size_frac = float(np.exp(log_size))
            direction = 1 if rng.random() < cfg.jump_inflow_prob_inst else -1
            flow_inst += direction * size_frac * aum_inst
        ind[day] = flow_ind
        inst[day] = flow_inst
        aum_ind  += flow_ind
        aum_inst += flow_inst
    return ind, inst


def _make_rng(session_seed: int, experiment_index: int) -> np.random.Generator:
    """Reproducible RNG for (session_seed, experiment_index)."""
    ss = np.random.SeedSequence(session_seed)
    child = ss.spawn(experiment_index + 1)[experiment_index]
    return np.random.default_rng(child)


def run_path(
    config: SimConfig,
    impparam: np.ndarray,           # shape (2,): (theta_ind, theta_inst)
    n_days: int,
    session_seed: int,
    experiment_index: int,
) -> tuple[
    np.ndarray,  # cash_series
    np.ndarray,  # invested_series
    np.ndarray,  # aum_ind_series
    np.ndarray,  # aum_inst_series
    np.ndarray,  # flow_ind_series
    np.ndarray,  # flow_inst_series
    np.ndarray,  # shortfall_ind_series   (deferred; small penalty)
    np.ndarray,  # shortfall_inst_series  (immediate; high penalty)
    list[JumpEvent],
]:
    """
    Simulate one fund path of length n_days.

    impparam: array-like of length 2, (theta_ind, theta_inst).
    """
    theta = np.asarray(impparam, dtype=float).reshape(-1)
    if theta.shape != (2,):
        raise ValueError(f"impparam must be shape (2,); got {theta.shape}")
    theta_ind, theta_inst = float(theta[0]), float(theta[1])

    rng = _make_rng(session_seed, experiment_index)
    cfg = config
    tdy = cfg.trading_days_per_year

    # --- Per-day parameters (converted from annual once) ---
    dt = 1.0 / tdy

    mu_ind_daily = cfg.mu_ind_annual * dt
    sigma_ind_daily = cfg.sigma_ind_annual * np.sqrt(dt)

    jump_rate_daily = cfg.jump_rate_inst_annual * dt
    r_market_daily = cfg.r_market_annual * dt
    mu_regime_daily = tuple(m * dt for m in cfg.mu_regime_annual)

    # --- Initial state ---
    aum_total = cfg.initial_aum
    aum_ind  = cfg.initial_aum_ind_fraction * aum_total
    aum_inst = aum_total - aum_ind
    cash = theta_ind * aum_ind + theta_inst * aum_inst
    invested = aum_total - cash
    regime = cfg.initial_regime

    # --- Output arrays ---
    cash_series           = np.empty(n_days)
    invested_series       = np.empty(n_days)
    aum_ind_series        = np.empty(n_days)
    aum_inst_series       = np.empty(n_days)
    flow_ind_series       = np.empty(n_days)
    flow_inst_series      = np.empty(n_days)
    shortfall_ind_series  = np.zeros(n_days)
    shortfall_inst_series = np.zeros(n_days)
    event_log: list[JumpEvent] = []

    for day in range(n_days):
        # 1. Regime transition (HMM)
        if not cfg.stationary and rng.random() < cfg.regime_switch_prob:
            regime = 1 - regime

        # 2. Individual (retail) net flow: GBM on aum_ind
        mu_eff = mu_ind_daily + mu_regime_daily[regime]
        flow_ind = aum_ind * (mu_eff + sigma_ind_daily * rng.standard_normal())

        # 3. Institutional jumps: Poisson count then lognormal sizes, on aum_inst
        n_jumps = rng.poisson(jump_rate_daily)
        flow_inst = 0.0
        for _ in range(n_jumps):
            log_size = cfg.jump_mean_log_inst + cfg.jump_std_log_inst * rng.standard_normal()
            size_frac = float(np.exp(log_size))
            direction = 1 if rng.random() < cfg.jump_inflow_prob_inst else -1
            flow_inst += direction * size_frac * aum_inst
            event_log.append(JumpEvent(day=day, size_fraction=size_frac, direction=direction))

        # 4. Apply flows: AUM tracks the deposit/withdrawal; cash receives it.
        aum_ind  += flow_ind
        aum_inst += flow_inst
        cash     += flow_ind + flow_inst

        # 5. Shortfall attribution (before rebalance).
        # Only if the day's cash went negative do we have any shortfall.
        if cash < 0.0:
            deficit = -cash
            # How much of the deficit came from the institutional outflow?
            inst_outflow = max(-flow_inst, 0.0)     # 0 if institutional inflow or none
            shortfall_inst = min(deficit, inst_outflow)
            shortfall_ind  = deficit - shortfall_inst

            # Institutional shortfall: immediate forced liquidation from invested.
            if shortfall_inst > 0.0:
                # Can only liquidate what's actually invested (no leverage).
                liquidation = min(invested, shortfall_inst)
                invested -= liquidation
                cash     += liquidation
                # If invested < shortfall_inst (rare, extreme), the remaining
                # shortfall still counts as institutional-caused for penalty.

            # Individual shortfall: deferred — cash allowed to stay negative
            # (represents temporarily-owed payments). It'll be repaid over
            # subsequent days by natural inflows or rebalance transfers.

            shortfall_inst_series[day] = shortfall_inst
            shortfall_ind_series[day]  = shortfall_ind

        # 6. Invested assets earn r_market.
        # Credit growth to aum_ind / aum_inst in proportion to their AUM shares,
        # so aum_ind + aum_inst tracks cash + invested (portfolio identity).
        growth = invested * r_market_daily
        invested += growth
        total_aum = aum_ind + aum_inst
        if total_aum > 0.0:
            aum_ind  += growth * (aum_ind  / total_aum)
            aum_inst += growth * (aum_inst / total_aum)

        # 7. End-of-day rebalance: target = theta_ind * aum_ind + theta_inst * aum_inst.
        target_cash = theta_ind * aum_ind + theta_inst * aum_inst
        gap = target_cash - cash
        transfer = cfg.rebalance_speed * gap
        # Can only take from invested (no leverage).
        transfer = max(transfer, -invested)
        cash     += transfer
        invested -= transfer

        # 8. Record end-of-day state
        cash_series[day]      = cash
        invested_series[day]  = invested
        aum_ind_series[day]   = aum_ind
        aum_inst_series[day]  = aum_inst
        flow_ind_series[day]  = flow_ind
        flow_inst_series[day] = flow_inst

    return (
        cash_series, invested_series,
        aum_ind_series, aum_inst_series,
        flow_ind_series, flow_inst_series,
        shortfall_ind_series, shortfall_inst_series,
        event_log,
    )
