from __future__ import annotations

import numpy as np

from .config import SimConfig
from .costs import compute_costs, compute_rewards
from .dynamics import run_path
from .result import SimResult


def simulate(
    config: SimConfig,
    impparam,                    # array-like of length 2: (theta_ind, theta_inst)
    horizon_weeks: int,
    session_seed: int,
    experiment_index: int,
) -> SimResult:
    """
    Pure, seedable simulation of one fund path.

    Same (config, impparam, horizon_weeks, session_seed, experiment_index)
    always returns a byte-identical SimResult.  Different experiment_index
    values draw independent paths, so observed costs differ even at fixed θ.

    impparam: 2-vector (theta_ind, theta_inst) — the individual/institutional
    buffer fractions.  Both components must lie in the box specified by
    config.impparam_min / config.impparam_max.
    """
    theta = np.asarray(impparam, dtype=float).reshape(-1)
    if theta.shape != (2,):
        raise ValueError(f"impparam must be shape (2,); got {theta.shape}")
    lo = np.asarray(config.impparam_min, dtype=float)
    hi = np.asarray(config.impparam_max, dtype=float)
    if np.any(theta < lo) or np.any(theta > hi):
        raise ValueError(
            f"impparam={theta.tolist()} outside box [{lo.tolist()}, {hi.tolist()}]"
        )

    n_days = horizon_weeks * 5  # 5 trading days per week

    (cash_series, invested_series,
     aum_ind_series, aum_inst_series,
     flow_ind_series, flow_inst_series,
     shortfall_ind_series, shortfall_inst_series,
     event_log) = run_path(
        config=config,
        impparam=theta,
        n_days=n_days,
        session_seed=session_seed,
        experiment_index=experiment_index,
    )

    opportunity_cost, shortfall_ind_cost, shortfall_inst_cost = compute_costs(
        config=config,
        impparam=theta,
        cash_series=cash_series,
        invested_series=invested_series,
        aum_ind_series=aum_ind_series,
        aum_inst_series=aum_inst_series,
        shortfall_ind_series=shortfall_ind_series,
        shortfall_inst_series=shortfall_inst_series,
    )
    total_cost = opportunity_cost + shortfall_ind_cost + shortfall_inst_cost

    market_gain, cash_gain, shortfall_ind_penalty, shortfall_inst_penalty = compute_rewards(
        config=config,
        impparam=theta,
        cash_series=cash_series,
        invested_series=invested_series,
        aum_ind_series=aum_ind_series,
        aum_inst_series=aum_inst_series,
        shortfall_ind_series=shortfall_ind_series,
        shortfall_inst_series=shortfall_inst_series,
    )
    total_reward = market_gain + cash_gain - shortfall_ind_penalty - shortfall_inst_penalty

    return SimResult(
        impparam=theta,
        session_seed=session_seed,
        experiment_index=experiment_index,
        total_reward=total_reward,
        market_gain=market_gain,
        cash_gain=cash_gain,
        shortfall_ind_penalty=shortfall_ind_penalty,
        shortfall_inst_penalty=shortfall_inst_penalty,
        total_cost=total_cost,
        opportunity_cost=opportunity_cost,
        shortfall_ind_cost=shortfall_ind_cost,
        shortfall_inst_cost=shortfall_inst_cost,
        cash_series=cash_series,
        invested_series=invested_series,
        aum_ind_series=aum_ind_series,
        aum_inst_series=aum_inst_series,
        flow_ind_series=flow_ind_series,
        flow_inst_series=flow_inst_series,
        shortfall_ind_series=shortfall_ind_series,
        shortfall_inst_series=shortfall_inst_series,
        event_log=event_log,
        days=n_days,
    )
