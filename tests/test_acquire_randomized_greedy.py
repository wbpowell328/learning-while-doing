"""
RandomizedGreedyPolicy tests.

The policy picks argmin of the belief's posterior mean over its grid
(the "best" θ in the internal minimise frame), then jitters that θ by
N(0, σ_greedy²), per-dim in the multivariate case, clipped to the box.
σ_greedy = 0 → pure greedy (deterministic argmin of the mean).
"""
import numpy as np

from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import AcquisitionConfig, RandomizedGreedyPolicy


def _loaded_model_1d() -> BeliefModel:
    # Cost-frame observations: minimum near θ=0.10.
    m = BeliefModel(BeliefConfig(length_scale=0.04, signal_std=3000.0,
                                 noise_std=15.0, prior_mean=0.0))
    for t, y in [(0.05, 2_000.0), (0.10, 500.0), (0.15, 2_500.0)]:
        m.update(t, y)
    return m


def test_greedy_returns_argmin_when_sigma_zero():
    cfg = AcquisitionConfig(sigma_greedy=0.0)
    policy = RandomizedGreedyPolicy(cfg)
    model = _loaded_model_1d()
    rng = np.random.default_rng(0)
    c = policy.propose(model, rng)
    # With sigma=0 every call should return the exact same θ.
    for seed in range(5):
        assert policy.propose(model, np.random.default_rng(seed)) == c
    # And it should be the grid point closest to the observed minimum.
    assert 0.08 <= c <= 0.12, f"greedy pick {c} not near the observed minimum at θ=0.10"


def test_sigma_greedy_perturbs_the_pick():
    cfg = AcquisitionConfig(sigma_greedy=0.02)
    policy = RandomizedGreedyPolicy(cfg)
    model = _loaded_model_1d()
    proposals = [policy.propose(model, np.random.default_rng(s)) for s in range(20)]
    # Not all identical (some noise applied).
    assert len(set(proposals)) > 1, "sigma_greedy > 0 should give varying proposals"
    # All inside the θ box.
    for p in proposals:
        assert 0.01 <= p <= 0.20, f"proposal {p} outside the θ box"


def test_clips_to_bounds_when_noise_pushes_out():
    # Tiny box + huge sigma → noise almost always falls outside.
    cfg = AcquisitionConfig(impparam_min=0.05, impparam_max=0.06, sigma_greedy=10.0)
    policy = RandomizedGreedyPolicy(cfg)
    model = _loaded_model_1d()
    for s in range(30):
        p = policy.propose(model, np.random.default_rng(s))
        assert 0.05 <= p <= 0.06, f"seed={s}: proposal {p} escaped clip"


def test_deterministic_given_same_rng():
    cfg = AcquisitionConfig(sigma_greedy=0.03)
    policy = RandomizedGreedyPolicy(cfg)
    model = _loaded_model_1d()
    a = policy.propose(model, np.random.default_rng(11))
    b = policy.propose(model, np.random.default_rng(11))
    assert a == b


def test_2d_perturbs_per_dimension():
    cfg = AcquisitionConfig(
        impparam_min=[0.01, 0.01], impparam_max=[0.20, 0.40],
        grid_size=25, sigma_greedy=0.02,
    )
    policy = RandomizedGreedyPolicy(cfg)
    model = BeliefModel(BeliefConfig(length_scale=[0.04, 0.08],
                                     signal_std=3000.0, noise_std=15.0,
                                     prior_mean=0.0), dim=2)
    model.update([0.10, 0.20], 500.0)
    model.update([0.05, 0.10], 2000.0)
    picks = np.stack([
        policy.propose(model, np.random.default_rng(s)) for s in range(20)
    ])
    # Both dims varied across picks.
    assert picks[:, 0].std() > 0
    assert picks[:, 1].std() > 0
    # Both dims stay inside the box.
    assert (picks[:, 0] >= 0.01).all() and (picks[:, 0] <= 0.20).all()
    assert (picks[:, 1] >= 0.01).all() and (picks[:, 1] <= 0.40).all()
