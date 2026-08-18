"""Test 9: edge cases — boundary θ, 1-week horizon, forced zero-jump path."""
import numpy as np
import pytest

from apps.cash_balance import SimConfig, simulate


CFG = SimConfig(stationary=True)


def test_impparam_at_minimum():
    """θ = impparam_min should run without error and return valid costs."""
    r = simulate(CFG, impparam=CFG.impparam_min, horizon_weeks=4,
                 session_seed=0, experiment_index=0)
    assert r.total_cost >= 0.0
    assert r.days == 20


def test_impparam_at_maximum():
    """θ = impparam_max should run without error and return valid costs."""
    r = simulate(CFG, impparam=CFG.impparam_max, horizon_weeks=4,
                 session_seed=0, experiment_index=0)
    assert r.total_cost >= 0.0
    assert r.days == 20


def test_impparam_out_of_range_raises():
    with pytest.raises(ValueError):
        simulate(CFG, impparam=0.0, horizon_weeks=4, session_seed=0, experiment_index=0)
    with pytest.raises(ValueError):
        simulate(CFG, impparam=0.5, horizon_weeks=4, session_seed=0, experiment_index=0)


def test_one_week_horizon():
    """1-week horizon produces exactly 5 trading days."""
    r = simulate(CFG, impparam=0.05, horizon_weeks=1, session_seed=0, experiment_index=0)
    assert r.days == 5
    assert len(r.cash_series) == 5
    assert len(r.flow_series) == 5


def test_zero_jump_path():
    """With jump_rate_annual=0, event_log must be empty."""
    cfg = SimConfig(stationary=True, jump_rate_annual=0.0)
    r = simulate(cfg, impparam=0.05, horizon_weeks=26, session_seed=0, experiment_index=0)
    assert r.event_log == [], f"Expected no jumps; got {len(r.event_log)}"


def test_zero_flow_no_shortfall():
    """Zero drift, zero vol, zero jumps → cash never goes negative."""
    cfg = SimConfig(
        stationary=True,
        mu_net_annual=0.0,
        sigma_net_annual=0.0,
        jump_rate_annual=0.0,
    )
    r = simulate(cfg, impparam=0.10, horizon_weeks=52, session_seed=0, experiment_index=0)
    assert r.shortfall_cost == 0.0
    assert np.all(r.cash_series >= 0.0)


def test_result_fields_consistent():
    """Basic shape consistency: all series have length == days."""
    r = simulate(CFG, impparam=0.08, horizon_weeks=8, session_seed=1, experiment_index=2)
    assert len(r.cash_series) == r.days
    assert len(r.invested_series) == r.days
    assert len(r.flow_series) == r.days
    assert len(r.shortfall_series) == r.days


def test_shortfall_series_nonnegative():
    """Pre-rebalance shortfall must never be negative."""
    r = simulate(CFG, impparam=0.01, horizon_weeks=26, session_seed=10, experiment_index=0)
    assert np.all(r.shortfall_series >= 0.0)


# ---------------------------------------------------------------------------
# Transaction noise (σ^trans): mean-0 $/day noise on profit only.
# ---------------------------------------------------------------------------
from dataclasses import replace


def test_transaction_noise_touches_profit_not_cash():
    """σ^trans perturbs total_reward only; the cash/flow path is unchanged,
    the noise is reported separately, and it's reproducible per seed."""
    quiet = replace(CFG, sigma_trans=0.0)
    noisy = replace(CFG, sigma_trans=750.0)
    a = simulate(quiet, 0.10, 0, 42, 0, n_days=100)
    b = simulate(noisy, 0.10, 0, 42, 0, n_days=100)
    assert np.array_equal(a.cash_series, b.cash_series)
    assert np.array_equal(a.flow_series, b.flow_series)
    assert a.transaction_noise == 0.0
    assert b.transaction_noise != 0.0
    assert b.total_reward == pytest.approx(
        b.market_gain + b.cash_gain - b.shortfall_penalty + b.transaction_noise
    )
    # Reproducible for the same (seed, experiment_index).
    assert simulate(noisy, 0.10, 0, 42, 0, n_days=100).transaction_noise == pytest.approx(
        b.transaction_noise
    )


def test_transaction_noise_mean_zero_std_inflated():
    """Across seeds the mean reward is ~unchanged (mean-0 noise) but the
    std is clearly inflated."""
    quiet = replace(CFG, sigma_trans=0.0)
    noisy = replace(CFG, sigma_trans=750.0)
    rq = np.array([simulate(quiet, 0.10, 0, s, 0, n_days=100).total_reward for s in range(300)])
    rn = np.array([simulate(noisy, 0.10, 0, s, 0, n_days=100).total_reward for s in range(300)])
    assert abs(rn.mean() - rq.mean()) < 0.15 * abs(rq.mean())
    assert rn.std() > 3 * rq.std()
