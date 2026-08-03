"""
Smoke + shape tests for the 2-parameter cash-balance simulator.
"""
import numpy as np
import pytest

from apps.cash_balance_2d import simulate, SimConfig


CFG = SimConfig(stationary=True)


def test_returns_scalar_positive_cost():
    r = simulate(CFG, [0.05, 0.10], horizon_weeks=26, session_seed=42, experiment_index=0)
    assert r.total_cost >= 0
    assert r.opportunity_cost >= 0
    assert r.shortfall_ind_cost >= 0
    assert r.shortfall_inst_cost >= 0
    assert np.isclose(
        r.total_cost,
        r.opportunity_cost + r.shortfall_ind_cost + r.shortfall_inst_cost,
    )


def test_theta_must_be_length_2():
    with pytest.raises(ValueError):
        simulate(CFG, [0.05], 26, 42, 0)             # scalar-like
    with pytest.raises(ValueError):
        simulate(CFG, [0.05, 0.10, 0.20], 26, 42, 0) # 3-vector


def test_theta_must_be_in_box():
    with pytest.raises(ValueError):
        simulate(CFG, [0.005, 0.10], 26, 42, 0)  # theta_ind below min
    with pytest.raises(ValueError):
        simulate(CFG, [0.10, 0.60], 26, 42, 0)   # theta_inst above max


def test_reproducibility_same_seed_same_cost():
    a = simulate(CFG, [0.05, 0.20], 26, 42, 0).total_cost
    b = simulate(CFG, [0.05, 0.20], 26, 42, 0).total_cost
    assert a == b


def test_crn_different_theta_shares_noise_stream():
    """Same (seed, exp_index), different theta — noise draws are shared."""
    r1 = simulate(CFG, [0.05, 0.20], 26, 42, 0)
    r2 = simulate(CFG, [0.10, 0.30], 26, 42, 0)
    # The Poisson counts / lognormal jump sizes / regime path are shared,
    # so the event_log should have jumps on the same days.
    days1 = sorted({e.day for e in r1.event_log})
    days2 = sorted({e.day for e in r2.event_log})
    assert days1 == days2


def test_low_theta_inst_causes_more_institutional_shortfalls():
    """A near-zero institutional buffer should produce visibly more forced-liquidation cost."""
    low  = [simulate(CFG, [0.05, 0.02], 26, 42, i).shortfall_inst_cost for i in range(20)]
    high = [simulate(CFG, [0.05, 0.30], 26, 42, i).shortfall_inst_cost for i in range(20)]
    assert np.mean(low) > np.mean(high), (
        f"expected more inst shortfall at theta_inst=0.02 ({np.mean(low):.1f}) "
        f"than at 0.30 ({np.mean(high):.1f})"
    )


def test_series_lengths_match_horizon():
    weeks = 10
    r = simulate(CFG, [0.05, 0.20], horizon_weeks=weeks, session_seed=42, experiment_index=0)
    expected = weeks * 5
    for name in ["cash_series", "invested_series",
                 "aum_ind_series", "aum_inst_series",
                 "flow_ind_series", "flow_inst_series",
                 "shortfall_ind_series", "shortfall_inst_series"]:
        arr = getattr(r, name)
        assert arr.shape == (expected,), f"{name} has shape {arr.shape}, expected ({expected},)"


def test_aum_and_portfolio_agree_at_end():
    """aum_ind + aum_inst should equal cash + invested (portfolio identity)."""
    r = simulate(CFG, [0.05, 0.20], horizon_weeks=26, session_seed=42, experiment_index=0)
    aum_sum = r.aum_ind_series[-1] + r.aum_inst_series[-1]
    portfolio = r.cash_series[-1] + r.invested_series[-1]
    # Tolerance because of the deferred-individual shortfall bookkeeping
    # (cash may be temporarily negative and reconciled over subsequent days).
    assert abs(aum_sum - portfolio) < 1e-6 * max(abs(aum_sum), 1.0), (
        f"AUM sum {aum_sum:.2f} != portfolio {portfolio:.2f}"
    )
