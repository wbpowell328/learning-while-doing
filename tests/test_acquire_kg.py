"""
KGPolicy (knowledge gradient) tests.

Properties verified:
  1. Proposal is always a valid grid point within [impparam_min, impparam_max].
  2. Works with no observations (prior only).
  3. KG values over the grid are non-negative (analytic KG, tight tolerance).
  4. The proposed point achieves the maximum KG value on the grid.
  5. Deterministic: same model → same proposal (regardless of rng seed).
  6. After dense observations in one region, KG prefers the unexplored region.
  7. KG values are higher in unexplored regions than in well-observed ones.
"""
import numpy as np
import pytest

from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import AcquisitionConfig, KGPolicy


CFG = AcquisitionConfig(impparam_min=0.01, impparam_max=0.20, grid_size=50)
BELIEF_CFG = BeliefConfig(length_scale=0.04, signal_std=5_000.0, noise_std=1_000.0)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_model(*observations) -> BeliefModel:
    m = BeliefModel(BELIEF_CFG)
    for c, cost in observations:
        m.update(c, cost)
    return m


def grid() -> np.ndarray:
    return np.linspace(CFG.impparam_min, CFG.impparam_max, CFG.grid_size)


# ---------------------------------------------------------------------------
# 1. Valid grid point
# ---------------------------------------------------------------------------

def test_proposal_is_grid_point():
    policy = KGPolicy(CFG)
    model = make_model((0.10, 3_000.0))
    proposal = policy.propose(model, make_rng(0))
    g = grid()
    assert any(np.isclose(proposal, g)), f"Proposal {proposal} not on evaluation grid"


def test_proposal_in_bounds():
    policy = KGPolicy(CFG)
    for obs in [[], [(0.05, 2_000.0)], [(0.05, 2_000.0), (0.15, 5_000.0)]]:
        c = policy.propose(make_model(*obs), make_rng(0))
        assert CFG.impparam_min <= c <= CFG.impparam_max


# ---------------------------------------------------------------------------
# 2. Works with no observations
# ---------------------------------------------------------------------------

def test_no_observations_valid():
    policy = KGPolicy(CFG)
    c = policy.propose(make_model(), make_rng(42))
    assert CFG.impparam_min <= c <= CFG.impparam_max


# ---------------------------------------------------------------------------
# 3. KG values are non-negative
# ---------------------------------------------------------------------------

def test_kg_values_non_negative():
    """
    KG ≥ 0 in theory.  Allow a small MC tolerance for sampling noise.
    """
    policy = KGPolicy(CFG)
    model = make_model((0.07, 2_000.0), (0.13, 4_500.0))
    kg = policy.kg_values(model, make_rng(7))
    # Analytic KG is exact: any negativity is float roundoff only.
    assert np.all(kg >= -1e-6), (
        f"KG has negative values beyond roundoff: min={kg.min():.4e}"
    )


def test_kg_values_non_negative_no_data():
    policy = KGPolicy(CFG)
    kg = policy.kg_values(make_model(), make_rng(7))
    assert np.all(kg >= -1e-6)


# ---------------------------------------------------------------------------
# 4. Proposed point achieves maximum KG
# ---------------------------------------------------------------------------

def test_proposes_max_kg_point():
    """propose() must return grid[argmax(kg_values())]."""
    policy = KGPolicy(CFG)
    model = make_model((0.10, 3_000.0))
    rng_a = make_rng(5)
    rng_b = make_rng(5)  # identical state
    proposal = policy.propose(model, rng_a)
    kg = policy.kg_values(model, rng_b)
    expected = grid()[np.argmax(kg)]
    assert np.isclose(proposal, expected), (
        f"propose() returned {proposal:.4f} but argmax(kg) is {expected:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Analytic KG: deterministic regardless of rng seed
# ---------------------------------------------------------------------------

def test_deterministic_across_seeds():
    """
    Analytic KG has no MC noise, so different rng seeds must all yield the
    same proposal (rng is accepted for interface compat but unused inside KG).
    """
    policy = KGPolicy(CFG)
    model = make_model((0.08, 2_500.0), (0.14, 4_000.0))
    proposals = {policy.propose(model, make_rng(s)) for s in range(10)}
    assert len(proposals) == 1, (
        f"Analytic KG should be deterministic but produced {len(proposals)} distinct proposals"
    )


# ---------------------------------------------------------------------------
# 7. Prefers unexplored region after dense observations
# ---------------------------------------------------------------------------

def test_prefers_unexplored_region():
    """
    After observing densely on the right half [0.12, 0.20], KG should
    direct the next query to the unexplored left half [0.01, 0.10].
    """
    policy = KGPolicy(CFG)
    m = BeliefModel(BeliefConfig(length_scale=0.03, signal_std=5_000.0, noise_std=200.0))
    for c in np.linspace(0.12, 0.20, 6):
        m.update(float(c), 5_000.0)
    proposal = policy.propose(m, make_rng(0))
    assert proposal < 0.11, (
        f"After dense right-half observations, expected proposal < 0.11, got {proposal:.4f}"
    )


# ---------------------------------------------------------------------------
# 8. KG higher in unexplored than well-observed regions
# ---------------------------------------------------------------------------

def test_kg_lower_at_observed_region():
    """
    Mean KG over the observed region [0.06, 0.10] should be lower than
    mean KG over the unobserved region [0.14, 0.18].
    """
    policy = KGPolicy(CFG)
    m = BeliefModel(BeliefConfig(length_scale=0.03, signal_std=5_000.0, noise_std=300.0))
    for c in np.linspace(0.06, 0.10, 5):
        m.update(float(c), 3_000.0)

    g = grid()
    kg = policy.kg_values(m, make_rng(0))

    observed_mask = (g >= 0.06) & (g <= 0.10)
    unexplored_mask = (g >= 0.14) & (g <= 0.18)

    mean_kg_observed = float(np.mean(kg[observed_mask]))
    mean_kg_unexplored = float(np.mean(kg[unexplored_mask]))

    assert mean_kg_observed < mean_kg_unexplored, (
        f"KG in observed region ({mean_kg_observed:.1f}) should be lower than "
        f"in unexplored region ({mean_kg_unexplored:.1f})"
    )
