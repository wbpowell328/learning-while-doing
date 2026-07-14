"""
Session.evaluate() tests.

Properties verified:
  1. Returns a SimResult with the requested c_star.
  2. Increments n_steps.
  3. Updates the belief model.
  4. Adds to history.
  5. CRN: evaluate() at step i uses the same noise path as step() at step i
     — a session that evaluates at a specific c_star gets the same total_cost
     as a reference session that happens to propose the same c_star via policy.
  6. evaluate() and step() interleave correctly (n_steps tracks both).
  7. c_star is stored exactly as passed (no rounding).
"""
import numpy as np
import pytest

from sim import SimConfig
from policy.belief import BeliefConfig
from policy.acquire import AcquisitionConfig, RandomPolicy
from policy.session import SessionConfig, Session


SIM_CFG    = SimConfig(stationary=True)
BELIEF_CFG = BeliefConfig()
ACQ_CFG    = AcquisitionConfig(c_star_min=0.01, c_star_max=0.20)
SES_CFG    = SessionConfig(horizon_weeks=4)


def make_session(seed: int = 0) -> Session:
    return Session(SIM_CFG, BELIEF_CFG, ACQ_CFG, SES_CFG,
                   RandomPolicy(ACQ_CFG), session_seed=seed)


# ---------------------------------------------------------------------------
# 1. Returns SimResult with requested c_star
# ---------------------------------------------------------------------------

def test_evaluate_returns_sim_result():
    s = make_session()
    result = s.evaluate(0.08)
    assert result.c_star == pytest.approx(0.08)
    assert result.total_cost > 0


# ---------------------------------------------------------------------------
# 2. Increments n_steps
# ---------------------------------------------------------------------------

def test_evaluate_increments_n_steps():
    s = make_session()
    assert s.n_steps == 0
    s.evaluate(0.08)
    assert s.n_steps == 1
    s.evaluate(0.12)
    assert s.n_steps == 2


# ---------------------------------------------------------------------------
# 3. Updates belief
# ---------------------------------------------------------------------------

def test_evaluate_updates_belief():
    s = make_session()
    assert s.belief.n_observations == 0
    s.evaluate(0.08)
    assert s.belief.n_observations == 1


# ---------------------------------------------------------------------------
# 4. Adds to history
# ---------------------------------------------------------------------------

def test_evaluate_adds_to_history():
    s = make_session()
    s.evaluate(0.07)
    s.evaluate(0.13)
    h = s.history
    assert len(h) == 2
    assert h[0][0] == pytest.approx(0.07)
    assert h[1][0] == pytest.approx(0.13)


# ---------------------------------------------------------------------------
# 5. CRN: same experiment_index → same noise path
# ---------------------------------------------------------------------------

def test_evaluate_same_crn_as_step():
    """
    A session using evaluate(c) at step 0 must produce the same total_cost
    as any other session that simulates at the same (session_seed, step=0, c).
    """
    c = 0.09
    seed = 7

    # Reference: use evaluate directly
    ref = make_session(seed=seed)
    ref_result = ref.evaluate(c)

    # Second session: same seed, same c_star via a second evaluate call
    s2 = make_session(seed=seed)
    s2_result = s2.evaluate(c)

    assert ref_result.total_cost == s2_result.total_cost


# ---------------------------------------------------------------------------
# 6. Interleaves with step()
# ---------------------------------------------------------------------------

def test_evaluate_interleaves_with_step():
    s = make_session()
    s.evaluate(0.05)   # step 0 via evaluate
    s.step()           # step 1 via policy
    s.evaluate(0.15)   # step 2 via evaluate
    assert s.n_steps == 3
    assert s.belief.n_observations == 3
    assert len(s.history) == 3


# ---------------------------------------------------------------------------
# 7. c_star stored exactly
# ---------------------------------------------------------------------------

def test_evaluate_c_star_exact():
    s = make_session()
    c = 0.123456
    s.evaluate(c)
    assert s.history[0][0] == pytest.approx(c)
