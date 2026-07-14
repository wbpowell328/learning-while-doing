"""
Belief model tests.

Properties verified:
  1. Prior (no data): mean = prior_mean, std = signal_std everywhere.
  2. Posterior std at an observed point < prior std (uncertainty shrinks).
  3. Posterior mean near a low-noise observation ≈ observed value.
  4. Posterior std increases as query moves away from the observed data.
  5. std is non-negative everywhere, including between observations.
  6. Determinism: identical observation sequences produce identical posteriors.
  7. n_observations and observations accessor track state correctly.
  8. posterior_at is consistent with posterior over an array.
  9. Two distant observations: posterior mean lies between them at the midpoint.
"""
import numpy as np
import pytest

from policy.belief import BeliefConfig, BeliefModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh(prior_mean=5_000.0, signal_std=5_000.0, noise_std=3_000.0,
          length_scale=0.04, **kw) -> BeliefModel:
    return BeliefModel(BeliefConfig(
        prior_mean=prior_mean, signal_std=signal_std,
        noise_std=noise_std, length_scale=length_scale, **kw
    ))


# ---------------------------------------------------------------------------
# 1. Prior
# ---------------------------------------------------------------------------

def test_prior_mean_uniform():
    model = fresh(prior_mean=7_000.0, signal_std=4_000.0)
    grid = np.linspace(0.01, 0.20, 30)
    mean, _ = model.posterior(grid)
    np.testing.assert_allclose(mean, 7_000.0)


def test_prior_std_uniform():
    model = fresh(signal_std=4_000.0)
    grid = np.linspace(0.01, 0.20, 30)
    _, std = model.posterior(grid)
    np.testing.assert_allclose(std, 4_000.0)


# ---------------------------------------------------------------------------
# 2. Posterior std shrinks at observed point
# ---------------------------------------------------------------------------

def test_std_shrinks_at_observation():
    model = fresh(noise_std=100.0)  # very low noise → tight posterior
    prior_std = model.config.signal_std
    model.update(0.10, 3_000.0)
    _, std = model.posterior(np.array([0.10]))
    assert std[0] < prior_std, (
        f"std at observed point ({std[0]:.1f}) should be below prior std ({prior_std:.1f})"
    )


def test_std_shrinks_with_more_data():
    """Adding a second nearby observation should further reduce uncertainty there."""
    model = fresh(noise_std=500.0, length_scale=0.05)
    model.update(0.10, 3_000.0)
    _, std_one = model.posterior(np.array([0.10]))
    model.update(0.10, 3_100.0)
    _, std_two = model.posterior(np.array([0.10]))
    assert std_two[0] < std_one[0], (
        f"std with two observations ({std_two[0]:.1f}) should be < "
        f"std with one ({std_one[0]:.1f})"
    )


# ---------------------------------------------------------------------------
# 3. Posterior mean tracks a low-noise observation
# ---------------------------------------------------------------------------

def test_posterior_mean_tracks_observation():
    """With very low observation noise, mean at the data point ≈ observed cost."""
    model = fresh(noise_std=1.0, prior_mean=0.0)
    model.update(0.10, 5_000.0)
    mean, _ = model.posterior(np.array([0.10]))
    assert abs(mean[0] - 5_000.0) < 50.0, (
        f"Posterior mean ({mean[0]:.1f}) should be close to 5000 with near-zero noise"
    )


# ---------------------------------------------------------------------------
# 4. Std increases with distance from data
# ---------------------------------------------------------------------------

def test_std_increases_away_from_data():
    model = fresh(length_scale=0.04)
    model.update(0.10, 3_000.0)
    _, std_near = model.posterior(np.array([0.10]))
    _, std_far = model.posterior(np.array([0.01]))  # 9 length-scales away
    assert std_far[0] > std_near[0], (
        f"std at 0.01 ({std_far[0]:.1f}) should exceed std at 0.10 ({std_near[0]:.1f})"
    )


def test_std_monotone_moving_away():
    """Std should increase monotonically as we move further from a single observation."""
    model = fresh(length_scale=0.04)
    model.update(0.10, 3_000.0)
    query = np.array([0.10, 0.11, 0.13, 0.16, 0.20])
    _, std = model.posterior(query)
    for i in range(len(std) - 1):
        assert std[i] <= std[i + 1] + 1e-9, (
            f"std not monotone: std[{query[i]:.2f}]={std[i]:.1f} > "
            f"std[{query[i+1]:.2f}]={std[i+1]:.1f}"
        )


# ---------------------------------------------------------------------------
# 5. std is non-negative everywhere
# ---------------------------------------------------------------------------

def test_std_non_negative_no_data():
    model = fresh()
    _, std = model.posterior(np.linspace(0.01, 0.20, 100))
    assert np.all(std >= 0.0)


def test_std_non_negative_with_data():
    model = fresh()
    for c, cost in [(0.05, 2_000.0), (0.10, 3_000.0), (0.15, 5_000.0)]:
        model.update(c, cost)
    _, std = model.posterior(np.linspace(0.01, 0.20, 200))
    assert np.all(std >= 0.0), f"Negative std encountered: min={std.min():.2e}"


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

def test_deterministic():
    observations = [(0.05, 2_000.0), (0.10, 3_500.0), (0.15, 4_800.0)]
    m1, m2 = fresh(), fresh()
    for c, cost in observations:
        m1.update(c, cost)
        m2.update(c, cost)
    grid = np.linspace(0.01, 0.20, 50)
    mean1, std1 = m1.posterior(grid)
    mean2, std2 = m2.posterior(grid)
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(std1, std2)


# ---------------------------------------------------------------------------
# 7. State tracking
# ---------------------------------------------------------------------------

def test_n_observations_increments():
    model = fresh()
    assert model.n_observations == 0
    model.update(0.05, 1_000.0)
    assert model.n_observations == 1
    model.update(0.10, 2_000.0)
    assert model.n_observations == 2


def test_observations_accessor_returns_copies():
    model = fresh()
    model.update(0.05, 1_000.0)
    model.update(0.10, 2_000.0)
    x, y = model.observations
    assert x == [0.05, 0.10]
    assert y == [1_000.0, 2_000.0]
    # mutating the returned lists must not affect the model
    x.append(0.99)
    y.append(99_999.0)
    assert model.n_observations == 2


# ---------------------------------------------------------------------------
# 8. posterior_at consistent with posterior
# ---------------------------------------------------------------------------

def test_posterior_at_consistent_with_posterior():
    model = fresh()
    model.update(0.10, 3_000.0)
    for c in [0.01, 0.08, 0.12, 0.20]:
        mean_arr, std_arr = model.posterior(np.array([c]))
        mean_pt, std_pt = model.posterior_at(c)
        assert abs(mean_arr[0] - mean_pt) < 1e-12
        assert abs(std_arr[0] - std_pt) < 1e-12


# ---------------------------------------------------------------------------
# 9. Interpolation between two observations
# ---------------------------------------------------------------------------

def test_mean_interpolates_between_observations():
    """
    With two observations bracketing a midpoint, the posterior mean at the
    midpoint should be between the two observed costs (not pulled to prior_mean
    or extrapolated beyond both values).
    """
    model = fresh(noise_std=10.0, prior_mean=0.0, length_scale=0.08)
    low_cost, high_cost = 1_000.0, 9_000.0
    model.update(0.04, low_cost)
    model.update(0.16, high_cost)
    mean_mid, _ = model.posterior(np.array([0.10]))
    assert low_cost < mean_mid[0] < high_cost, (
        f"Midpoint posterior mean ({mean_mid[0]:.1f}) should be between "
        f"{low_cost} and {high_cost}"
    )
