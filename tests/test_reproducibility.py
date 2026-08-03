"""Tests 1 & 2: determinism and noise."""
import numpy as np
import pytest

from sim import SimConfig, simulate


CFG = SimConfig(stationary=True)


def test_same_inputs_identical_result():
    """Test 1: same (seed, index, params) → byte-identical output."""
    r1 = simulate(CFG, c_star=0.05, horizon_weeks=4, session_seed=42, experiment_index=0)
    r2 = simulate(CFG, c_star=0.05, horizon_weeks=4, session_seed=42, experiment_index=0)
    assert r1.total_cost == r2.total_cost
    np.testing.assert_array_equal(r1.cash_series, r2.cash_series)
    np.testing.assert_array_equal(r1.invested_series, r2.invested_series)
    assert r1.event_log == r2.event_log


def test_different_experiment_index_different_cost():
    """Test 2: same θ, different experiment_index → different costs (noise exists)."""
    costs = {
        simulate(CFG, c_star=0.05, horizon_weeks=52, session_seed=42, experiment_index=i).total_cost
        for i in range(20)
    }
    # With 20 independent draws the probability all are equal is astronomically small
    assert len(costs) > 1, "All 20 experiments returned identical cost — noise is missing"


def test_different_seed_different_result():
    """Bonus: different session_seed at same experiment_index → different path."""
    r1 = simulate(CFG, c_star=0.05, horizon_weeks=4, session_seed=1, experiment_index=0)
    r2 = simulate(CFG, c_star=0.05, horizon_weeks=4, session_seed=2, experiment_index=0)
    assert not np.array_equal(r1.cash_series, r2.cash_series)
