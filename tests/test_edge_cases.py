"""Test 9: edge cases — boundary C*, 1-week horizon, forced zero-jump path."""
import numpy as np
import pytest

from sim import SimConfig, simulate


CFG = SimConfig(stationary=True)


def test_c_star_at_minimum():
    """C* = c_star_min should run without error and return valid costs."""
    r = simulate(CFG, c_star=CFG.c_star_min, horizon_weeks=4,
                 session_seed=0, experiment_index=0)
    assert r.total_cost >= 0.0
    assert r.days == 20


def test_c_star_at_maximum():
    """C* = c_star_max should run without error and return valid costs."""
    r = simulate(CFG, c_star=CFG.c_star_max, horizon_weeks=4,
                 session_seed=0, experiment_index=0)
    assert r.total_cost >= 0.0
    assert r.days == 20


def test_c_star_out_of_range_raises():
    with pytest.raises(ValueError):
        simulate(CFG, c_star=0.0, horizon_weeks=4, session_seed=0, experiment_index=0)
    with pytest.raises(ValueError):
        simulate(CFG, c_star=0.5, horizon_weeks=4, session_seed=0, experiment_index=0)


def test_one_week_horizon():
    """1-week horizon produces exactly 5 trading days."""
    r = simulate(CFG, c_star=0.05, horizon_weeks=1, session_seed=0, experiment_index=0)
    assert r.days == 5
    assert len(r.cash_series) == 5
    assert len(r.flow_series) == 5


def test_zero_jump_path():
    """With jump_rate_annual=0, event_log must be empty."""
    cfg = SimConfig(stationary=True, jump_rate_annual=0.0)
    r = simulate(cfg, c_star=0.05, horizon_weeks=26, session_seed=0, experiment_index=0)
    assert r.event_log == [], f"Expected no jumps; got {len(r.event_log)}"


def test_zero_flow_no_shortfall():
    """Zero drift, zero vol, zero jumps → cash never goes negative."""
    cfg = SimConfig(
        stationary=True,
        mu_net_annual=0.0,
        sigma_net_annual=0.0,
        jump_rate_annual=0.0,
    )
    r = simulate(cfg, c_star=0.10, horizon_weeks=52, session_seed=0, experiment_index=0)
    assert r.shortfall_cost == 0.0
    assert np.all(r.cash_series >= 0.0)


def test_result_fields_consistent():
    """Basic shape consistency: all series have length == days."""
    r = simulate(CFG, c_star=0.08, horizon_weeks=8, session_seed=1, experiment_index=2)
    assert len(r.cash_series) == r.days
    assert len(r.invested_series) == r.days
    assert len(r.flow_series) == r.days
    assert len(r.shortfall_series) == r.days


def test_shortfall_series_nonnegative():
    """Pre-rebalance shortfall must never be negative."""
    r = simulate(CFG, c_star=0.01, horizon_weeks=26, session_seed=10, experiment_index=0)
    assert np.all(r.shortfall_series >= 0.0)
