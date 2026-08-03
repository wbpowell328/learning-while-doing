from __future__ import annotations

from .config import SimConfig
from .costs import compute_costs
from .dynamics import run_path
from .result import SimResult


def simulate(
    config: SimConfig,
    c_star: float,
    horizon_weeks: int,
    session_seed: int,
    experiment_index: int,
) -> SimResult:
    """
    Pure, seedable simulation of one fund path.

    Same (config, c_star, horizon_weeks, session_seed, experiment_index)
    always returns a byte-identical SimResult.  Different experiment_index
    values draw independent paths, so observed costs differ even at fixed θ.
    """
    if not (config.c_star_min <= c_star <= config.c_star_max):
        raise ValueError(
            f"c_star={c_star} outside [{config.c_star_min}, {config.c_star_max}]"
        )

    n_days = horizon_weeks * 5  # 5 trading days per week

    cash_series, invested_series, flow_series, shortfall_series, event_log = run_path(
        config=config,
        c_star=c_star,
        n_days=n_days,
        session_seed=session_seed,
        experiment_index=experiment_index,
    )

    opportunity_cost, shortfall_cost = compute_costs(
        config=config,
        c_star=c_star,
        cash_series=cash_series,
        invested_series=invested_series,
        shortfall_series=shortfall_series,
    )

    total_cost = opportunity_cost + shortfall_cost

    return SimResult(
        c_star=c_star,
        session_seed=session_seed,
        experiment_index=experiment_index,
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
