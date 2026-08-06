"""
GET /sessions/{sid}/flow_sample — deposits/redemptions sample-path
report. Returns per-day non-negative arrays split by investor class
(individual vs institutional) and direction (deposit vs redemption).
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)


def _make(app_name: str = "cash_balance") -> str:
    r = client.post("/sessions", json={
        "app_name": app_name, "policy": "kg", "session_seed": 42,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


def test_flow_sample_1d_shape_and_signs():
    sid = _make()
    r = client.get(f"/sessions/{sid}/flow_sample?horizon=30")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon"] == 30
    assert body["dim"] == 1
    assert len(body["days"]) == 30
    for k in ("individual_deposit", "individual_redemption",
              "institutional_deposit", "institutional_redemption"):
        assert len(body[k]) == 30
        assert all(v >= 0 for v in body[k]), f"{k} has a negative value"
    # Deposits and redemptions never both non-zero on the same day for
    # a given investor class — they're the split of a signed net flow.
    for a, b in zip(body["individual_deposit"], body["individual_redemption"]):
        assert not (a > 0 and b > 0), "individual deposit + redemption both > 0 on same day"


def test_flow_sample_2d():
    sid = _make("cash_balance_2d")
    r = client.get(f"/sessions/{sid}/flow_sample?horizon=15")
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 15
    assert body["dim"] == 2
    assert len(body["days"]) == 15
    for k in ("individual_deposit", "individual_redemption",
              "institutional_deposit", "institutional_redemption"):
        assert len(body[k]) == 15


def test_flow_sample_clamps_extreme_horizon():
    """Typo protection — horizon capped to [1, 5000]."""
    sid = _make()
    r = client.get(f"/sessions/{sid}/flow_sample?horizon=999999")
    assert r.status_code == 200
    assert r.json()["horizon"] == 5000
    r = client.get(f"/sessions/{sid}/flow_sample?horizon=-5")
    assert r.status_code == 200
    assert r.json()["horizon"] == 1


def test_flow_sample_deterministic_across_calls():
    """Same session_seed → same sample path."""
    sid = _make()
    a = client.get(f"/sessions/{sid}/flow_sample?horizon=10").json()
    b = client.get(f"/sessions/{sid}/flow_sample?horizon=10").json()
    assert a == b
