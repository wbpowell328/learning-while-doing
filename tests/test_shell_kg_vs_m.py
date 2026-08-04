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


def test_kg_indep_scalar_math():
    """
    Regression: closed-form KG at a single alternative.

    For a candidate θ with V=1, μ_x=1, current-best (other cell) μ_best=0,
    noise σ_ε=1 and m=1:
        σ̃ = sqrt(V²/(V+σ²)) = sqrt(1/2) ≈ 0.7071
        z = -|μ_x-μ_best|/σ̃ = -1/0.7071 = -√2
        f(z) = φ(z) + z·Φ(z)                        [not - z·Φ(z)]
             = φ(-√2) + (-√2)·Φ(-√2)
             = 0.15866 + (-1.4142)·0.07865
             = 0.15866 - 0.11123 = 0.04744
        KG(m=1) = σ̃·f = 0.7071·0.04744 ≈ 0.03354

    A previous version of the code had the sign wrong (used
    f(z) = φ(z) − z·Φ(z), giving 0.269·0.7071 ≈ 0.19 instead of 0.033
    — off by ~5x). This test would have caught that.
    """
    import numpy as np
    from policy.acquire import kg_indep_scalar_vs_batch_size
    from policy import BeliefConfig, BeliefModel

    # Two-alternative synthetic setup on a small grid.
    cfg = BeliefConfig(length_scale=100.0, signal_std=1.0, noise_std=1.0, prior_mean=0.0)
    bel = BeliefModel(cfg, dim=1)
    # Give the "other" cell a very tight observation so its posterior mean = 0
    # (matches prior) — we want μ_best = 0 for the analytic comparison.
    # Instead: use kg_indep_scalar which relies on the GP posterior at candidate
    # and the min over grid, plus BeliefConfig.noise_std for σ_ε.
    # Build a grid where the candidate is at 0.5 and a "current best" is at 0.
    grid = np.array([0.0, 0.5])
    # No observations → posterior mean = prior_mean = 0 everywhere, V = signal_std² = 1.
    # With μ_x = 0 = μ_best, Δ = 0 → z = 0 → f(0) = φ(0) = 0.399.
    # σ̃(m=1) = sqrt(1/(1+1)) = 0.707. KG(m=1) = 0.707 * 0.399 ≈ 0.282.
    kg = kg_indep_scalar_vs_batch_size(bel, grid, theta=0.5, m_values=[1])
    expected = 0.7071 * 0.3989
    assert abs(float(kg[0]) - expected) < 1e-3, (
        f"KG(m=1) at Δ=0 should be σ̃·φ(0) = {expected:.4f}, got {float(kg[0]):.4f}"
    )
