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
