"""Test 3: θ drives the realized cash trajectory, not just the score."""
import numpy as np
import pytest

from apps.cash_balance import SimConfig, simulate


CFG = SimConfig(stationary=True)


def test_different_impparam_different_cash_series():
    """
    Two runs with the SAME seed/index but DIFFERENT θ must produce
    materially different cash trajectories.  This guards against a regression
    where θ only re-scales the score at the end.
    """
    r_low  = simulate(CFG, impparam=0.02, horizon_weeks=26, session_seed=7, experiment_index=0)
    r_high = simulate(CFG, impparam=0.15, horizon_weeks=26, session_seed=7, experiment_index=0)

    # Cash series must differ
    assert not np.allclose(r_low.cash_series, r_high.cash_series), (
        "Cash series is identical for impparam=0.02 and impparam=0.15 — "
        "θ is not driving the trajectory"
    )


def test_higher_impparam_higher_average_cash():
    """
    On average over many paths, higher θ should produce higher cash balances
    (modulo flows).  This is the mechanical consequence of targeting a higher buffer.
    """
    seeds = range(30)
    mean_cash_low  = np.mean([
        simulate(CFG, impparam=0.02, horizon_weeks=26, session_seed=s, experiment_index=0).cash_series.mean()
        for s in seeds
    ])
    mean_cash_high = np.mean([
        simulate(CFG, impparam=0.15, horizon_weeks=26, session_seed=s, experiment_index=0).cash_series.mean()
        for s in seeds
    ])
    assert mean_cash_high > mean_cash_low, (
        f"Expected mean cash to be higher for impparam=0.15 than 0.02; "
        f"got {mean_cash_high:.0f} vs {mean_cash_low:.0f}"
    )
