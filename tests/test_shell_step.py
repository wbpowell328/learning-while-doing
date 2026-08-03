"""
POST /sessions/{sid}/step tests.

Properties verified:
  1. Returns 200 with expected fields.
  2. impparam is within [impparam_min, impparam_max].
  3. total_cost is positive.
  4. days matches horizon_weeks * 5.
  5. n_steps increments with each call.
  6. best_impparam is within [impparam_min, impparam_max].
  7. Stepping an unknown session returns 404.
  8. Reproducible: two sessions with same seed produce identical step results.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)

ACQ_MIN = 0.01
ACQ_MAX = 0.20
HORIZON = 4  # weeks → 20 days


def make_session(policy: str = "random", seed: int = 42) -> str:
    r = client.post("/sessions", json={
        "sim_config": {"stationary": True},
        "session_config": {"horizon_weeks": HORIZON},
        "policy": policy,
        "session_seed": seed,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


# ---------------------------------------------------------------------------
# 1. Response shape
# ---------------------------------------------------------------------------

def test_step_response_fields():
    sid = make_session()
    r = client.post(f"/sessions/{sid}/step")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {"impparam", "total_cost", "days", "n_steps", "best_impparam"}


# ---------------------------------------------------------------------------
# 2. impparam in bounds
# ---------------------------------------------------------------------------

def test_step_impparam_in_bounds():
    sid = make_session()
    for _ in range(5):
        data = client.post(f"/sessions/{sid}/step").json()
        assert ACQ_MIN <= data["impparam"] <= ACQ_MAX


# ---------------------------------------------------------------------------
# 3. total_cost positive
# ---------------------------------------------------------------------------

def test_step_total_cost_positive():
    sid = make_session()
    data = client.post(f"/sessions/{sid}/step").json()
    assert data["total_cost"] > 0.0


# ---------------------------------------------------------------------------
# 4. days matches horizon
# ---------------------------------------------------------------------------

def test_step_days_matches_horizon():
    sid = make_session()
    data = client.post(f"/sessions/{sid}/step").json()
    assert data["days"] == HORIZON * 5


# ---------------------------------------------------------------------------
# 5. n_steps increments
# ---------------------------------------------------------------------------

def test_step_n_steps_increments():
    sid = make_session()
    for expected in range(1, 4):
        data = client.post(f"/sessions/{sid}/step").json()
        assert data["n_steps"] == expected


# ---------------------------------------------------------------------------
# 6. best_impparam in bounds
# ---------------------------------------------------------------------------

def test_step_best_impparam_in_bounds():
    sid = make_session()
    for _ in range(3):
        data = client.post(f"/sessions/{sid}/step").json()
        assert ACQ_MIN <= data["best_impparam"] <= ACQ_MAX


# ---------------------------------------------------------------------------
# 7. Unknown session → 404
# ---------------------------------------------------------------------------

def test_step_unknown_session():
    r = client.post("/sessions/does-not-exist/step")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 8. Reproducibility
# ---------------------------------------------------------------------------

def test_step_reproducible():
    sid_a = make_session(seed=7)
    sid_b = make_session(seed=7)
    for _ in range(3):
        a = client.post(f"/sessions/{sid_a}/step").json()
        b = client.post(f"/sessions/{sid_b}/step").json()
        assert a["impparam"] == b["impparam"]
        assert a["total_cost"] == b["total_cost"]
