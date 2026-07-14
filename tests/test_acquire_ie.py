"""
IEPolicy (information-theoretic exploration) tests.

Properties verified:
  1. Proposal is always a grid point within [c_star_min, c_star_max].
  2. The proposed point achieves the maximum posterior std on the grid.
  3. After observing a point, the policy does NOT re-propose that point
     (std is lowest there, so argmax std moves elsewhere).
  4. Deterministic: rng is not used; same model → same proposal.
  5. Works with an empty model (prior is flat; first proposal is grid[0]).
  6. More observations shift the proposal toward unexplored regions.
"""
import numpy as np
import pytest

from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import AcquisitionConfig, IEPolicy


CFG = AcquisitionConfig(c_star_min=0.01, c_star_max=0.20, grid_size=100)
BELIEF_CFG = BeliefConfig(length_scale=0.04, signal_std=5_000.0, noise_std=500.0)

_DUMMY_RNG = np.random.default_rng(0)  # IE never uses rng; pass any instance


def make_model(*observations) -> BeliefModel:
    m = BeliefModel(BELIEF_CFG)
    for c, cost in observations:
        m.update(c, cost)
    return m


def grid() -> np.ndarray:
    return np.linspace(CFG.c_star_min, CFG.c_star_max, CFG.grid_size)


# ---------------------------------------------------------------------------
# 1. Proposal is a valid grid point
# ---------------------------------------------------------------------------

def test_proposal_is_grid_point():
    policy = IEPolicy(CFG)
    model = make_model((0.10, 3_000.0))
    proposal = policy.propose(model, _DUMMY_RNG)
    g = grid()
    assert any(np.isclose(proposal, g)), (
        f"Proposal {proposal} is not on the evaluation grid"
    )


def test_proposal_in_bounds():
    policy = IEPolicy(CFG)
    for model in [make_model(), make_model((0.05, 2_000.0), (0.15, 5_000.0))]:
        c = policy.propose(model, _DUMMY_RNG)
        assert CFG.c_star_min <= c <= CFG.c_star_max


# ---------------------------------------------------------------------------
# 2. Proposed point has maximum posterior std
# ---------------------------------------------------------------------------

def test_proposes_max_std_point():
    policy = IEPolicy(CFG)
    model = make_model((0.10, 3_000.0), (0.12, 4_000.0))
    g = grid()
    _, std = model.posterior(g)
    proposal = policy.propose(model, _DUMMY_RNG)
    # The proposed point must achieve the maximum std (up to floating-point equality)
    _, proposal_std = model.posterior(np.array([proposal]))
    assert np.isclose(proposal_std[0], np.max(std)), (
        f"Proposal std {proposal_std[0]:.1f} ≠ max std {np.max(std):.1f}"
    )


# ---------------------------------------------------------------------------
# 3. Does not re-propose a well-observed point
# ---------------------------------------------------------------------------

def test_avoids_observed_point():
    """Low noise → posterior std at 0.10 is very small; IE should avoid it."""
    policy = IEPolicy(CFG)
    # Observe many times at the same point to make uncertainty there near zero
    m = BeliefModel(BeliefConfig(noise_std=50.0))
    for _ in range(5):
        m.update(0.10, 3_000.0)
    proposal = policy.propose(m, _DUMMY_RNG)
    # The proposal must NOT be at the saturated point
    assert not np.isclose(proposal, 0.10, atol=0.005), (
        f"IE re-proposed heavily-observed point 0.10 (got {proposal:.4f})"
    )


# ---------------------------------------------------------------------------
# 4. Deterministic (rng not used)
# ---------------------------------------------------------------------------

def test_deterministic_ignores_rng():
    policy = IEPolicy(CFG)
    model = make_model((0.08, 2_500.0))
    # Two different rng instances — result must be identical
    c1 = policy.propose(model, np.random.default_rng(1))
    c2 = policy.propose(model, np.random.default_rng(99))
    assert c1 == c2, "IEPolicy result changed with different rng — it should be deterministic"


def test_deterministic_same_model():
    policy = IEPolicy(CFG)
    model = make_model((0.05, 2_000.0), (0.15, 5_000.0))
    c1 = policy.propose(model, _DUMMY_RNG)
    c2 = policy.propose(model, _DUMMY_RNG)
    assert c1 == c2


# ---------------------------------------------------------------------------
# 5. Empty model returns a valid grid point
# ---------------------------------------------------------------------------

def test_empty_model_returns_grid_point():
    policy = IEPolicy(CFG)
    model = make_model()
    c = policy.propose(model, _DUMMY_RNG)
    g = grid()
    assert any(np.isclose(c, g)), f"Empty-model proposal {c} not on grid"


# ---------------------------------------------------------------------------
# 6. More observations shift proposals toward unexplored regions
# ---------------------------------------------------------------------------

def test_explores_after_dense_left_observations():
    """
    After many observations on the left half [0.01, 0.09], IE should
    shift its proposal toward the right (unexplored) half [0.10, 0.20].
    """
    policy = IEPolicy(CFG)
    m = BeliefModel(BeliefConfig(length_scale=0.03, noise_std=200.0))
    for c in np.linspace(0.01, 0.09, 8):
        m.update(float(c), 3_000.0)
    proposal = policy.propose(m, _DUMMY_RNG)
    assert proposal > 0.10, (
        f"After dense left-half observations, expected proposal > 0.10, got {proposal:.4f}"
    )
