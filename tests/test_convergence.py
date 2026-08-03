"""
Test 8: Monte Carlo convergence — variance of mean cost shrinks ~1/n.

We estimate Var[mean(cost, n)] at two sample sizes n1 < n2 and check
that the ratio is approximately n2/n1 (CLT).  We use a loose bound (2×)
to avoid a flaky test while still catching gross failures.
"""
import numpy as np
import pytest

from sim import SimConfig, simulate


CFG = SimConfig(stationary=True)
C_STAR = 0.08
SEED = 123
HORIZON = 26  # weeks


def _batch_costs(n: int, seed_offset: int = 0) -> np.ndarray:
    return np.array([
        simulate(CFG, impparam=C_STAR, horizon_weeks=HORIZON,
                 session_seed=SEED + seed_offset + i, experiment_index=0).total_cost
        for i in range(n)
    ])


def test_variance_shrinks_with_n():
    """
    Estimate Var[mean_n] via bootstrap or direct formula, for n=25 and n=100.
    Expected ratio of variances ≈ 4 (= 100/25).  Accept [2, 8] to avoid flakiness.
    """
    n1, n2 = 25, 100
    costs1 = _batch_costs(n1, seed_offset=0)
    costs2 = _batch_costs(n2, seed_offset=1000)

    # Var[sample mean] = sample_var / n
    var_mean1 = np.var(costs1, ddof=1) / n1
    var_mean2 = np.var(costs2, ddof=1) / n2

    ratio = var_mean1 / var_mean2  # expect ≈ n2/n1 = 4
    assert 2.0 <= ratio <= 8.0, (
        f"Variance ratio var_mean(n=25)/var_mean(n=100) = {ratio:.2f}; "
        f"expected ≈ 4.0 (CLT convergence).  "
        f"Var(n=25)={var_mean1:.4f}, Var(n=100)={var_mean2:.4f}"
    )
