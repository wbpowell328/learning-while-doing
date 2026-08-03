from __future__ import annotations

from .config import SimConfig
from .costs import compute_costs, compute_rewards
from .dynamics import run_path
from .result import SimResult


def simulate(
    config: SimConfig,
    impparam: float,
    horizon_weeks: int,
    session_seed: int,
    experiment_index: int,
) -> SimResult:
    """
    Pure, seedable simulation of one fund path.

    Same (config, impparam, horizon_weeks, session_seed, experiment_index)
    always returns a byte-identical SimResult.  Different experiment_index
    values draw independent paths, so observed costs differ even at fixed θ.
    """
    if not (config.impparam_min <= impparam <= config.impparam_max):
        raise ValueError(
            f"impparam={impparam} outside [{config.impparam_min}, {config.impparam_max}]"
        )

    n_days = horizon_weeks * 5  # 5 trading days per week

    cash_series, invested_series, flow_series, shortfall_series, event_log = run_path(
        config=config,
        impparam=impparam,
        n_days=n_days,
        session_seed=session_seed,
        experiment_index=experiment_index,
    )

    opportunity_cost, shortfall_cost = compute_costs(
        config=config,
        impparam=impparam,
        cash_series=cash_series,
        invested_series=invested_series,
        shortfall_series=shortfall_series,
    )
    total_cost = opportunity_cost + shortfall_cost

    market_gain, cash_gain, shortfall_penalty = compute_rewards(
        config=config,
        impparam=impparam,
        cash_series=cash_series,
        invested_series=invested_series,
        shortfall_series=shortfall_series,
    )
    total_reward = market_gain + cash_gain - shortfall_penalty

    return SimResult(
        impparam=impparam,
        session_seed=session_seed,
        experiment_index=experiment_index,
        total_reward=total_reward,
        market_gain=market_gain,
        cash_gain=cash_gain,
        shortfall_penalty=shortfall_penalty,
        total_cost=total_cost,
        opportunity_cost=opportunity_cost,
        shortfall_cost=shortfall_cost,
        cash_series=cash_series,
        invested_series=invested_series,
        flow_series=flow_series,
        shortfall_series=shortfall_series,
        event_log=event_log,
        days=n_days,
    )
