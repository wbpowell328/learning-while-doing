"""
Test 4: U-shaped cost curve.

Estimating F(θ) by averaging many seeds should yield:
  - high cost at very low θ (shortfall-dominated)
  - high cost at very high θ (drag-dominated)
  - a minimum somewhere in between
"""
import numpy as np
import pytest

from sim import SimConfig, simulate


CFG = SimConfig(stationary=True, jump_rate_annual=24.0)  # more jumps → clearer U


def _mean_cost(c_star: float, n: int = 80, seed_offset: int = 0) -> float:
    return np.mean([
        simulate(CFG, c_star=c_star, horizon_weeks=52, session_seed=s + seed_offset, experiment_index=0).total_cost
        for s in range(n)
    ])


def test_u_shape():
    """
    F(0.01) > F(mid) and F(0.20) > F(mid) for some mid in (0.04, 0.16).
    We test this at three grid points: low, mid, high.
    """
    cost_low  = _mean_cost(0.01)
    cost_mid  = _mean_cost(0.08)
    cost_high = _mean_cost(0.20)

    assert cost_low > cost_mid, (
        f"Expected cost at θ=0.01 ({cost_low:.2f}) > cost at θ=0.08 ({cost_mid:.2f}); "
        "shortfall cost should dominate at very low θ"
    )
    assert cost_high > cost_mid, (
        f"Expected cost at θ=0.20 ({cost_high:.2f}) > cost at θ=0.08 ({cost_mid:.2f}); "
        "opportunity cost (cash drag) should dominate at very high θ"
    )


def test_low_c_star_has_shortfall():
    """At very low θ, shortfall cost should be non-trivial (not zero)."""
    results = [
        simulate(CFG, c_star=0.01, horizon_weeks=52, session_seed=s, experiment_index=0)
        for s in range(40)
    ]
    mean_shortfall = np.mean([r.shortfall_cost for r in results])
    assert mean_shortfall > 0, "Expected non-zero shortfall at θ=0.01"


def test_high_c_star_has_drag():
    """At very high θ, opportunity cost should be non-trivial."""
    results = [
        simulate(CFG, c_star=0.20, horizon_weeks=52, session_seed=s, experiment_index=0)
        for s in range(40)
    ]
    mean_drag = np.mean([r.opportunity_cost for r in results])
    assert mean_drag > 0, "Expected non-zero opportunity cost at θ=0.20"
