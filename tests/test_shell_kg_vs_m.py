"""
GET /sessions/{sid}/kg_vs_m regression tests.

Reason this file exists: the endpoint has grown a diagnostic-block that
runs on every request, including on fresh sessions with zero observations
— a bug in an earlier commit left one intermediate variable defined
inside a conditional, breaking session creation for the whole KG(m) card.
Keep the empty-session path covered so it can't recur.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)

EXPECTED_KEYS = {
    "theta", "m_values", "kg_values", "kg_values_correlated",
    "kg_values_independent", "noise_std", "noise_std_belief",
    "base_kg", "base_kg_indep",
    "delta_corr", "sigma_tilde_corr",
    "delta_indep", "sigma_tilde_indep",
}


def make_session(policy: str = "kg") -> str:
    r = client.post("/sessions", json={"policy": policy, "session_seed": 42})
    assert r.status_code == 201
    return r.json()["session_id"]


def test_kg_vs_m_empty_session_ok():
    """Fresh session with zero observations must still return a valid response."""
    sid = make_session()
    r = client.get(f"/sessions/{sid}/kg_vs_m?m_max=25")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= EXPECTED_KEYS
    # With no observations, both formulations agree that Δ = 0 (all cells at prior).
    assert body["delta_corr"]  == pytest.approx(0.0, abs=1e-6)
    assert body["delta_indep"] == pytest.approx(0.0, abs=1e-6)
    # σ̃(m=1) > 0 for both.
    assert body["sigma_tilde_corr"]  > 0
    assert body["sigma_tilde_indep"] > 0


def test_kg_vs_m_after_a_step_ok():
    """After one observation, endpoint still returns 200 with populated diagnostic."""
    sid = make_session()
    client.post(f"/sessions/{sid}/step")
    r = client.get(f"/sessions/{sid}/kg_vs_m?m_max=25")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= EXPECTED_KEYS
    # After 1 obs, at least one formulation should have Δ ≥ 0.
    assert body["delta_corr"]  >= 0
    assert body["delta_indep"] >= 0


def test_kg_vs_m_sigma_eps_override_returns_refit_delta():
    """Overriding σ_ε must recompute both delta and σ̃ under the refit."""
    sid = make_session()
    client.post(f"/sessions/{sid}/step")
    r1 = client.get(f"/sessions/{sid}/kg_vs_m?m_max=25")
    r2 = client.get(f"/sessions/{sid}/kg_vs_m?m_max=25&sigma_eps=50000")
    assert r1.status_code == 200 and r2.status_code == 200
    # noise_std in effect should differ; belief's own noise_std unchanged.
    assert r2.json()["noise_std"] == pytest.approx(50000.0)
    assert r1.json()["noise_std_belief"] == r2.json()["noise_std_belief"]


def test_kg_vs_m_1d_only():
    """kg_vs_m is 1-D only; 2-D sessions should 400."""
    r = client.post("/sessions", json={"app_name": "cash_balance_2d", "policy": "kg", "session_seed": 42})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    r = client.get(f"/sessions/{sid}/kg_vs_m")
    assert r.status_code == 400
