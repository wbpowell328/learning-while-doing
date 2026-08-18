from __future__ import annotations

import numpy as np

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
    n_days: int | None = None,
) -> SimResult:
    """
    Pure, seedable simulation of one fund path.

    Same (config, impparam, horizon_weeks, session_seed, experiment_index)
    always returns a byte-identical SimResult.  Different experiment_index
    values draw independent paths, so observed costs differ even at fixed θ.

    `n_days` (optional) overrides the horizon_weeks×5 calculation — used by
    the /experiment endpoint where the user specifies days directly.
    """
    if not (config.impparam_min <= impparam <= config.impparam_max):
        raise ValueError(
            f"impparam={impparam} outside [{config.impparam_min}, {config.impparam_max}]"
        )

    if n_days is None:
        n_days = horizon_weeks * 5  # 5 trading days per week
    n_days = int(n_days)

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

    # Transaction noise: independent, mean-0 $/day noise on the PROFIT only.
    # One draw per day (so it scales with the horizon like the belief's
    # per-day noise), summed into the reward. Drawn from a separate RNG
    # stream keyed on (session_seed, experiment_index) so it's reproducible
    # and does NOT perturb the cash/flow path draws. The cash app maximises,
    # so the belief learns on total_reward (as −reward); total_cost is the
    # legacy cost-frame decomposition and is left untouched (identity holds).
    sigma_trans = float(getattr(config, "sigma_trans", 0.0) or 0.0)
    transaction_noise = 0.0
    if sigma_trans > 0.0:
        noise_rng = np.random.default_rng(
            np.random.SeedSequence(entropy=session_seed, spawn_key=(0x7A2C5, experiment_index))
        )
        transaction_noise = float(np.sum(noise_rng.normal(0.0, sigma_trans, size=n_days)))

    total_reward = market_gain + cash_gain - shortfall_penalty + transaction_noise

    return SimResult(
        impparam=impparam,
        session_seed=session_seed,
        experiment_index=experiment_index,
        total_reward=total_reward,
        market_gain=market_gain,
        cash_gain=cash_gain,
        shortfall_penalty=shortfall_penalty,
        transaction_noise=transaction_noise,
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
