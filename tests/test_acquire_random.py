"""
RandomPolicy tests.

Properties verified:
  1. Proposal is always within [impparam_min, impparam_max].
  2. Proposals vary across calls (not constant — noise exists).
  3. Same rng state → identical proposal (deterministic).
  4. Works with an empty model (no observations required).
  5. Works with a loaded model (model is ignored, but must not crash).
  6. Custom domain bounds are respected.
"""
import numpy as np
import pytest

from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import AcquisitionConfig, RandomPolicy


CFG = AcquisitionConfig(impparam_min=0.01, impparam_max=0.20)


def make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def empty_model() -> BeliefModel:
    return BeliefModel()


def loaded_model() -> BeliefModel:
    m = BeliefModel()
    for c, cost in [(0.05, 2_000.0), (0.10, 3_500.0), (0.15, 5_000.0)]:
        m.update(c, cost)
    return m


# ---------------------------------------------------------------------------
# 1. In-bounds
# ---------------------------------------------------------------------------

def test_proposal_in_bounds():
    policy = RandomPolicy(CFG)
    model = empty_model()
    for seed in range(50):
        c = policy.propose(model, make_rng(seed))
        assert CFG.impparam_min <= c <= CFG.impparam_max, (
            f"seed={seed}: proposal {c} outside [{CFG.impparam_min}, {CFG.impparam_max}]"
        )


# ---------------------------------------------------------------------------
# 2. Varies across calls
# ---------------------------------------------------------------------------

def test_proposals_vary():
    policy = RandomPolicy(CFG)
    model = empty_model()
    proposals = {policy.propose(model, make_rng(s)) for s in range(30)}
    assert len(proposals) > 1, "All 30 proposals are identical — randomness is absent"


# ---------------------------------------------------------------------------
# 3. Deterministic given same rng state
# ---------------------------------------------------------------------------

def test_deterministic():
    policy = RandomPolicy(CFG)
    model = empty_model()
    c1 = policy.propose(model, make_rng(7))
    c2 = policy.propose(model, make_rng(7))
    assert c1 == c2


# ---------------------------------------------------------------------------
# 4 & 5. Works with empty and loaded model
# ---------------------------------------------------------------------------

def test_works_with_empty_model():
    policy = RandomPolicy(CFG)
    c = policy.propose(empty_model(), make_rng(0))
    assert CFG.impparam_min <= c <= CFG.impparam_max


def test_works_with_loaded_model():
    policy = RandomPolicy(CFG)
    c = policy.propose(loaded_model(), make_rng(0))
    assert CFG.impparam_min <= c <= CFG.impparam_max


# ---------------------------------------------------------------------------
# 6. Custom bounds
# ---------------------------------------------------------------------------

def test_custom_bounds_respected():
    cfg = AcquisitionConfig(impparam_min=0.05, impparam_max=0.10)
    policy = RandomPolicy(cfg)
    model = empty_model()
    for seed in range(40):
        c = policy.propose(model, make_rng(seed))
        assert 0.05 <= c <= 0.10, f"seed={seed}: {c} outside [0.05, 0.10]"
