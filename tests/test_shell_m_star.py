"""
m* (batch-size trick) tests.

Backend contract:
* CreateSessionRequest accepts optional m_star (default 1); response echoes it.
* POST /sessions/{sid}/m_star updates the KG policy's precision multiplier.
* Non-KG policies (random, ie, human) accept the update as a no-op (they
  don't have set_m_star, but the session's own m_star still tracks it).
* The value flows into policy._m_star and, through kg_pre_compute, divides
  the noise variance in the KG closed form. Higher m_star means smaller
  effective noise, so KG values grow strictly.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app, _sessions

client = TestClient(app)


def test_m_star_default_is_1():
    r = client.post("/sessions", json={"policy": "kg", "session_seed": 42})
    assert r.status_code == 201
    assert r.json()["m_star"] == 1
    sess = _sessions[r.json()["session_id"]]
    assert sess.m_star == 1
    assert sess._policy._m_star == 1


def test_m_star_set_at_creation():
    r = client.post("/sessions", json={"policy": "kg", "session_seed": 42, "m_star": 5})
    assert r.status_code == 201
    assert r.json()["m_star"] == 5
    assert _sessions[r.json()["session_id"]]._policy._m_star == 5


def test_m_star_update_endpoint_kg_policy():
    r = client.post("/sessions", json={"policy": "kg", "session_seed": 42})
    sid = r.json()["session_id"]
    r2 = client.post(f"/sessions/{sid}/m_star", json={"m_star": 10})
    assert r2.status_code == 200
    assert r2.json() == {"m_star": 10}
    # Effective at the policy layer.
    assert _sessions[sid]._policy._m_star == 10


def test_m_star_update_non_kg_policy_is_noop_but_ok():
    """Random policy doesn't have set_m_star; endpoint still 200s."""
    r = client.post("/sessions", json={"policy": "random", "session_seed": 42})
    sid = r.json()["session_id"]
    r2 = client.post(f"/sessions/{sid}/m_star", json={"m_star": 7})
    assert r2.status_code == 200
    assert r2.json()["m_star"] == 7
    # Session tracks it even though no KG policy uses it.
    assert _sessions[sid].m_star == 7


def test_m_star_scales_policy_kg_values():
    """
    Regression on the ONE thing this feature is supposed to do: divide
    noise variance by m_star in the KG computation THAT THE POLICY USES.
    Higher m_star ⇒ smaller σ̃ ⇒ larger w_j ⇒ strictly larger KG per θ.

    Note: the KG(m) chart's `kg_vs_batch_size` is a SEPARATE hypothetical-
    batch computation that doesn't consume the policy's m_star; that's by
    design (the chart asks "what if we ran m obs at this θ?"). This test
    goes straight at the policy path — kg_analytic_correlated_at with
    different m_star values on the same belief.
    """
    import numpy as np
    from policy import BeliefConfig, BeliefModel, kg_analytic_correlated_at

    cfg = BeliefConfig(length_scale=0.04, signal_std=3000.0,
                       noise_std=1500.0, prior_mean=-35000.0)
    bel = BeliefModel(cfg, dim=1)
    # A handful of observations so KG has structure to work with.
    for theta, y in [(0.05, -36000), (0.10, -34500), (0.15, -35500)]:
        bel.update(theta, y)
    grid = np.linspace(0.01, 0.20, 40)
    kg_m1  = kg_analytic_correlated_at(bel, grid, grid, m_star=1)
    kg_m10 = kg_analytic_correlated_at(bel, grid, grid, m_star=10)
    # Cell-by-cell: m_star=10 must be ≥ m_star=1 everywhere, and strictly
    # greater somewhere.
    assert (kg_m10 >= kg_m1 - 1e-9).all(), (
        "m_star=10 should never give a smaller KG than m_star=1 at any θ"
    )
    assert (kg_m10 > kg_m1 + 1e-6).any(), (
        f"m_star=10 should raise KG somewhere: max diff = {(kg_m10 - kg_m1).max()}"
    )


def test_z_alpha_endpoint_updates_session():
    """POST /z_alpha stores the new value on acq_config; IE picks it up."""
    r = client.post("/sessions", json={"policy": "ie", "session_seed": 42})
    sid = r.json()["session_id"]
    assert _sessions[sid]._acq_config.z_alpha == 0.0

    r = client.post(f"/sessions/{sid}/z_alpha", json={"z_alpha": 2.5})
    assert r.status_code == 200
    assert r.json()["z_alpha"] == pytest.approx(2.5)
    assert _sessions[sid]._acq_config.z_alpha == pytest.approx(2.5)


def test_sigma_greedy_endpoint_updates_session():
    """POST /sigma_greedy stores the new value on acq_config."""
    r = client.post("/sessions", json={"policy": "randomized_greedy", "session_seed": 42})
    sid = r.json()["session_id"]
    assert _sessions[sid]._acq_config.sigma_greedy == 0.0

    r = client.post(f"/sessions/{sid}/sigma_greedy", json={"sigma_greedy": 0.03})
    assert r.status_code == 200
    assert r.json()["sigma_greedy"] == pytest.approx(0.03)
    assert _sessions[sid]._acq_config.sigma_greedy == pytest.approx(0.03)


def test_z_alpha_endpoint_on_non_ie_policy_is_stored():
    """Storing z_alpha on a KG session works; takes effect if user later swaps to IE."""
    r = client.post("/sessions", json={"policy": "kg", "session_seed": 42})
    sid = r.json()["session_id"]
    client.post(f"/sessions/{sid}/z_alpha", json={"z_alpha": 1.5})
    assert _sessions[sid]._acq_config.z_alpha == pytest.approx(1.5)


def test_kg_endpoint_reflects_session_m_star():
    """
    The /sessions/{sid}/kg endpoint feeds the KG(x) chart on the frontend.
    It must respect the session's current m_star so the noise-factor editor
    on that chart actually changes what the user sees. Post an observation
    so the KG has structure, snapshot the curve at m*=1, bump m*, snapshot
    again, and check the analytic KG values grew monotonically per probe.
    """
    r = client.post("/sessions", json={"policy": "kg", "session_seed": 42})
    sid = r.json()["session_id"]
    # Give the belief a couple of observations so KG > 0 somewhere.
    client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 2,
    })

    base = client.get(f"/sessions/{sid}/kg").json()
    client.post(f"/sessions/{sid}/m_star", json={"m_star": 10})
    boosted = client.get(f"/sessions/{sid}/kg").json()

    assert boosted["impparams"] == base["impparams"]
    for a, b in zip(base["analytic_correlated"], boosted["analytic_correlated"]):
        assert b + 1e-9 >= a, f"m*=10 must not lower KG at any probe (a={a}, b={b})"
    diffs = [b - a for a, b in zip(base["analytic_correlated"],
                                   boosted["analytic_correlated"])]
    assert max(diffs) > 1e-6, (
        "m*=10 must raise KG somewhere on the probe grid; the endpoint is "
        "not passing m_star to kg_analytic_correlated_at."
    )
