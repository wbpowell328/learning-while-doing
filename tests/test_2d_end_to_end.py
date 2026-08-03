"""
End-to-end integration tests for the 2-parameter pipeline:
BeliefModel (ARD) → policies (KG/IE/Random) on 2-D grid → Session dispatching
to cash_balance_2d.simulate → best_impparam retrieval.
"""
import numpy as np
import pytest

from apps.cash_balance_2d import simulate as sim_2d, SimConfig as SimConfig2D
from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import (
    AcquisitionConfig, RandomPolicy, IEPolicy, KGPolicy,
    _make_grid,
)
from policy.session import Session, SessionConfig


ACQ_2D = AcquisitionConfig(
    impparam_min=(0.01, 0.01),
    impparam_max=(0.20, 0.40),
    grid_size=15,       # 225 grid points — enough for a test
    z_alpha=0.0,
)
BEL_2D = BeliefConfig(
    length_scale=(0.03, 0.06),   # ARD: different per dim
    signal_std=3000.0,
    noise_std=800.0,
    prior_mean=3000.0,
)
SIM_CFG = SimConfig2D(stationary=True)
SES_CFG = SessionConfig(horizon_weeks=26, best_impparam_grid=15)


def _session_2d(policy):
    return Session(
        sim_config=SIM_CFG,
        belief_config=BEL_2D,
        acq_config=ACQ_2D,
        session_config=SES_CFG,
        policy=policy,
        session_seed=42,
        simulate_fn=sim_2d,
    )


def test_belief_ard_kernel_produces_directional_smoothness():
    """
    ARD with length_scale=(0.02, 0.20): correlation should drop fast in dim 0
    (short length scale) and slowly in dim 1 (long length scale).
    """
    m = BeliefModel(BeliefConfig(length_scale=(0.02, 0.20), signal_std=1.0), dim=2)
    m.update([0.10, 0.10], 5.0)
    # Query at points shifted in dim-0 vs shifted in dim-1 by the same amount.
    mu_shift0, _ = m.posterior([[0.15, 0.10]])   # 0.05 shift in dim 0 (~2.5 length scales)
    mu_shift1, _ = m.posterior([[0.10, 0.15]])   # 0.05 shift in dim 1 (~0.25 length scales)
    prior = m.config.prior_mean
    # Dim-1 shift should still be strongly informed (mu close to obs=5.0);
    # Dim-0 shift should be much closer to the prior mean.
    assert abs(mu_shift1[0] - 5.0) < abs(mu_shift0[0] - 5.0), (
        f"expected slower falloff in the long-length-scale dim: "
        f"shift0={mu_shift0[0]:.3f}, shift1={mu_shift1[0]:.3f}, prior={prior}"
    )


def test_grid_shape_matches_dim():
    grid = _make_grid((0.01, 0.01), (0.20, 0.40), grid_size=5)
    assert grid.shape == (25, 2)


def test_random_policy_returns_2vector_in_box():
    policy = RandomPolicy(ACQ_2D)
    rng = np.random.default_rng(0)
    for _ in range(10):
        theta = policy.propose(BeliefModel(BEL_2D, dim=2), rng)
        theta = np.asarray(theta)
        assert theta.shape == (2,)
        assert 0.01 <= theta[0] <= 0.20
        assert 0.01 <= theta[1] <= 0.40


def test_ie_policy_returns_2vector_in_box():
    policy = IEPolicy(ACQ_2D)
    rng = np.random.default_rng(0)
    model = BeliefModel(BEL_2D, dim=2)
    model.update([0.05, 0.20], 2500.0)
    theta = np.asarray(policy.propose(model, rng))
    assert theta.shape == (2,)
    assert 0.01 <= theta[0] <= 0.20
    assert 0.01 <= theta[1] <= 0.40


def test_kg_policy_returns_2vector_in_box():
    policy = KGPolicy(ACQ_2D)
    rng = np.random.default_rng(0)
    model = BeliefModel(BEL_2D, dim=2)
    model.update([0.05, 0.20], 2500.0)
    theta = np.asarray(policy.propose(model, rng))
    assert theta.shape == (2,)
    assert 0.01 <= theta[0] <= 0.20
    assert 0.01 <= theta[1] <= 0.40


def test_session_2d_step_dispatches_to_2d_sim():
    session = _session_2d(RandomPolicy(ACQ_2D))
    result = session.step()
    # cash_balance_2d SimResult has these fields; cash_balance (1D) doesn't.
    assert hasattr(result, "shortfall_ind_cost")
    assert hasattr(result, "shortfall_inst_cost")
    assert hasattr(result, "aum_ind_series")


def test_session_2d_best_impparam_is_2vector():
    session = _session_2d(KGPolicy(ACQ_2D))
    for _ in range(4):
        session.step()
    best = np.asarray(session.best_impparam())
    assert best.shape == (2,)


def test_session_2d_history_records_vector_theta():
    session = _session_2d(RandomPolicy(ACQ_2D))
    for _ in range(3):
        session.step()
    assert len(session.history) == 3
    for theta, cost in session.history:
        theta_arr = np.asarray(theta)
        assert theta_arr.shape == (2,)
        assert cost > 0


def test_kg_prefers_extreme_regions_early():
    """
    With a uniform prior, KG's argmax is often on the boundary of the box
    (where the correlation cascade is asymmetric). Just verify KG's first
    pick is at a corner or edge of the grid, not the interior.
    """
    policy = KGPolicy(ACQ_2D)
    model = BeliefModel(BEL_2D, dim=2)
    rng = np.random.default_rng(0)
    theta = np.asarray(policy.propose(model, rng))
    on_edge = (
        np.isclose(theta[0], 0.01) or np.isclose(theta[0], 0.20) or
        np.isclose(theta[1], 0.01) or np.isclose(theta[1], 0.40)
    )
    assert on_edge, f"expected KG's first pick on the boundary; got {theta}"
