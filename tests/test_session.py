"""
Session integration tests.

Properties verified:
  1.  Initial state: n_steps=0, empty belief, empty history.
  2.  step() increments n_steps.
  3.  step() returns a SimResult with the proposed impparam.
  4.  step() impparam is always within [impparam_min, impparam_max].
  5.  step() updates the belief model (n_observations grows).
  6.  Multiple steps accumulate correctly.
  7.  observe() adds to history but does NOT increment n_steps.
  8.  observe() updates the belief model.
  9.  history is returned as an independent copy.
  10. best_impparam() returns domain midpoint when no data.
  11. best_impparam() is in [impparam_min, impparam_max] after steps.
  12. best_impparam() tracks the minimum of injected observations.
  13. Full reproducibility: same seed → identical history.
  14. CRN: step i at any impparam uses the same noise path — verified by
      comparing two sessions whose only difference is policy (and hence
      proposed impparam); the same experiment_index uses the same underlying
      random draws.
"""
import numpy as np
import pytest

from apps.cash_balance import SimConfig
from policy.belief import BeliefConfig
from policy.acquire import AcquisitionConfig, RandomPolicy, IEPolicy
from policy.session import SessionConfig, Session


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIM_CFG = SimConfig(stationary=True, jump_rate_annual=12.0)
BELIEF_CFG = BeliefConfig(length_scale=0.04, signal_std=5_000.0, noise_std=3_000.0)
ACQ_CFG = AcquisitionConfig(impparam_min=0.01, impparam_max=0.20, grid_size=50)
SES_CFG = SessionConfig(horizon_weeks=4, best_impparam_grid=100)


def make_session(policy=None, seed: int = 42) -> Session:
    if policy is None:
        policy = RandomPolicy(ACQ_CFG)
    return Session(SIM_CFG, BELIEF_CFG, ACQ_CFG, SES_CFG, policy, session_seed=seed)


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

def test_initial_n_steps_zero():
    session = make_session()
    assert session.n_steps == 0


def test_initial_belief_empty():
    session = make_session()
    assert session.belief.n_observations == 0


def test_initial_history_empty():
    session = make_session()
    assert session.history == []


# ---------------------------------------------------------------------------
# 2. step() increments n_steps
# ---------------------------------------------------------------------------

def test_step_increments_n_steps():
    session = make_session()
    session.step()
    assert session.n_steps == 1
    session.step()
    assert session.n_steps == 2


# ---------------------------------------------------------------------------
# 3. step() returns SimResult with the proposed impparam
# ---------------------------------------------------------------------------

def test_step_returns_sim_result():
    session = make_session()
    result = session.step()
    assert ACQ_CFG.impparam_min <= result.impparam <= ACQ_CFG.impparam_max
    assert result.total_cost > 0.0
    assert result.days == SES_CFG.horizon_weeks * 5


# ---------------------------------------------------------------------------
# 4. impparam is in domain
# ---------------------------------------------------------------------------

def test_step_impparam_in_bounds():
    session = make_session()
    for _ in range(10):
        r = session.step()
        assert ACQ_CFG.impparam_min <= r.impparam <= ACQ_CFG.impparam_max, (
            f"impparam={r.impparam} outside [{ACQ_CFG.impparam_min}, {ACQ_CFG.impparam_max}]"
        )


# ---------------------------------------------------------------------------
# 5. step() updates belief
# ---------------------------------------------------------------------------

def test_step_updates_belief():
    session = make_session()
    session.step()
    assert session.belief.n_observations == 1
    session.step()
    assert session.belief.n_observations == 2


# ---------------------------------------------------------------------------
# 6. Multiple steps accumulate correctly
# ---------------------------------------------------------------------------

def test_multiple_steps():
    session = make_session()
    n = 5
    for _ in range(n):
        session.step()
    assert session.n_steps == n
    assert session.belief.n_observations == n
    assert len(session.history) == n


def test_history_matches_belief_observations():
    session = make_session()
    for _ in range(4):
        session.step()
    h_c, h_cost = zip(*session.history)
    b_c, b_cost = session.belief.observations
    assert list(h_c) == b_c
    assert list(h_cost) == b_cost


# ---------------------------------------------------------------------------
# 7. observe() does not increment n_steps
# ---------------------------------------------------------------------------

def test_observe_does_not_increment_n_steps():
    session = make_session()
    session.observe(0.10, 3_000.0)
    assert session.n_steps == 0


def test_observe_adds_to_history():
    session = make_session()
    session.observe(0.10, 3_000.0)
    h = session.history
    assert len(h) == 1
    assert h[0] == (0.10, 3_000.0)


def test_observe_and_step_interleave():
    session = make_session()
    session.observe(0.05, 2_000.0)
    session.step()
    session.observe(0.15, 5_000.0)
    session.step()
    assert session.n_steps == 2
    assert len(session.history) == 4
    assert session.belief.n_observations == 4


# ---------------------------------------------------------------------------
# 8. observe() updates belief
# ---------------------------------------------------------------------------

def test_observe_updates_belief():
    session = make_session()
    assert session.belief.n_observations == 0
    session.observe(0.10, 3_000.0)
    assert session.belief.n_observations == 1
    mean, std = session.belief.posterior_at(0.10)
    prior_std = session.belief.config.signal_std
    assert std < prior_std, "Std at observed point should drop below prior std"


# ---------------------------------------------------------------------------
# 9. history is an independent copy
# ---------------------------------------------------------------------------

def test_history_is_copy():
    session = make_session()
    session.step()
    h = session.history
    h.append((0.99, 99_999.0))
    assert len(session.history) == 1, "Mutating returned history should not affect session"


# ---------------------------------------------------------------------------
# 10. best_impparam() with no data returns midpoint
# ---------------------------------------------------------------------------

def test_best_impparam_no_data_returns_midpoint():
    session = make_session()
    mid = (ACQ_CFG.impparam_min + ACQ_CFG.impparam_max) / 2.0
    assert session.best_impparam() == mid


# ---------------------------------------------------------------------------
# 11. best_impparam() is in domain after steps
# ---------------------------------------------------------------------------

def test_best_impparam_in_bounds():
    session = make_session()
    for _ in range(6):
        session.step()
    best = session.best_impparam()
    assert ACQ_CFG.impparam_min <= best <= ACQ_CFG.impparam_max


# ---------------------------------------------------------------------------
# 12. best_impparam() tracks the minimum of injected observations
# ---------------------------------------------------------------------------

def test_best_impparam_tracks_injected_minimum():
    """
    Inject a parabola with minimum at 0.10.  The posterior mean should
    be lowest near 0.10, so best_impparam() should land close to it.
    """
    session = make_session()
    for c in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
        # cost is low near 0.10 and rises toward the extremes
        cost = 5_000.0 + 200_000.0 * (c - 0.10) ** 2
        session.observe(c, cost)
    best = session.best_impparam()
    assert abs(best - 0.10) < 0.04, (
        f"best_impparam={best:.4f} should be near 0.10 given the injected parabola"
    )


# ---------------------------------------------------------------------------
# 13. Full reproducibility
# ---------------------------------------------------------------------------

def test_reproducible():
    """Same seed + same policy type → identical sequence of (impparam, cost)."""
    n = 5
    s1, s2 = make_session(seed=7), make_session(seed=7)
    for _ in range(n):
        r1, r2 = s1.step(), s2.step()
        assert r1.impparam == r2.impparam, "impparam differs across identical sessions"
        assert r1.total_cost == r2.total_cost, "cost differs across identical sessions"
    assert s1.history == s2.history


def test_different_seed_produces_different_history():
    s1, s2 = make_session(seed=1), make_session(seed=2)
    for _ in range(5):
        s1.step()
        s2.step()
    # With overwhelmingly high probability, at least one step differs
    assert s1.history != s2.history


# ---------------------------------------------------------------------------
# 14. CRN: experiment_index is step-count-based, not affected by observe()
# ---------------------------------------------------------------------------

def test_crn_observe_does_not_shift_experiment_index():
    """
    observe() must not shift the experiment_index used by subsequent step()
    calls.  Inject observations between steps and verify the simulation
    outcomes still match a reference session with no observe() calls.
    """
    ref = make_session(seed=55)
    ref_r0 = ref.step()
    ref_r1 = ref.step()

    # Session with observe() injected before each step
    inj = make_session(seed=55)
    inj.observe(0.10, 3_000.0)   # should not shift experiment_index
    inj_r0 = inj.step()          # should use experiment_index=0
    inj.observe(0.12, 4_000.0)
    inj_r1 = inj.step()          # should use experiment_index=1

    assert inj_r0.impparam == ref_r0.impparam, (
        "observe() shifted the proposal for step 0"
    )
    assert inj_r0.total_cost == ref_r0.total_cost, (
        "observe() shifted the simulation seed for step 0"
    )
    assert inj_r1.impparam == ref_r1.impparam
    assert inj_r1.total_cost == ref_r1.total_cost
