"""
POST /sessions/{sid}/evaluate tests.

Properties verified:
  1. Returns 200 with expected fields.
  2. impparam in response matches the requested value.
  3. n_steps increments after each call.
  4. best_impparam is in domain.
  5. total_cost is positive.
  6. Interleaves correctly with step().
  7. Unknown session returns 404.
  8. Missing body returns 422.
  9. "human" policy sessions work end-to-end via evaluate.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)

ACQ_MIN = 0.01
ACQ_MAX = 0.20


def make_session(policy: str = "human") -> str:
    r = client.post("/sessions", json={
        "sim_config": {"stationary": True},
        "session_config": {"horizon_weeks": 4},
        "policy": policy,
        "session_seed": 42,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


def evaluate(sid: str, impparam: float) -> dict:
    r = client.post(f"/sessions/{sid}/evaluate", json={"impparam": impparam})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Response shape
# ---------------------------------------------------------------------------

def test_evaluate_response_fields():
    sid = make_session()
    data = evaluate(sid, 0.08)
    assert set(data.keys()) >= {"impparam", "total_cost", "days", "n_steps", "best_impparam"}


# ---------------------------------------------------------------------------
# 2. impparam echoed back
# ---------------------------------------------------------------------------

def test_evaluate_impparam_echoed():
    sid = make_session()
    data = evaluate(sid, 0.07)
    assert abs(data["impparam"] - 0.07) < 1e-9


# ---------------------------------------------------------------------------
# 3. n_steps increments
# ---------------------------------------------------------------------------

def test_evaluate_n_steps_increments():
    sid = make_session()
    for expected in range(1, 4):
        data = evaluate(sid, 0.10)
        assert data["n_steps"] == expected


# ---------------------------------------------------------------------------
# 4. best_impparam in domain
# ---------------------------------------------------------------------------

def test_evaluate_best_impparam_in_domain():
    sid = make_session()
    data = evaluate(sid, 0.10)
    assert ACQ_MIN <= data["best_impparam"] <= ACQ_MAX


# ---------------------------------------------------------------------------
# 5. total_cost positive
# ---------------------------------------------------------------------------

def test_evaluate_total_cost_positive():
    sid = make_session()
    data = evaluate(sid, 0.10)
    assert data["total_cost"] > 0


# ---------------------------------------------------------------------------
# 6. Interleaves with step()
# ---------------------------------------------------------------------------

def test_evaluate_interleaves_with_step():
    sid = make_session(policy="random")
    evaluate(sid, 0.05)
    client.post(f"/sessions/{sid}/step")
    evaluate(sid, 0.15)
    state = client.get(f"/sessions/{sid}/state").json()
    assert state["n_steps"] == 3
    assert state["n_observations"] == 3


# ---------------------------------------------------------------------------
# 7. Unknown session → 404
# ---------------------------------------------------------------------------

def test_evaluate_unknown_session():
    r = client.post("/sessions/ghost/evaluate", json={"impparam": 0.10})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 8. Missing body → 422
# ---------------------------------------------------------------------------

def test_evaluate_missing_body():
    sid = make_session()
    r = client.post(f"/sessions/{sid}/evaluate", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 9. "human" policy accepted at session creation
# ---------------------------------------------------------------------------

def test_human_policy_session_end_to_end():
    r = client.post("/sessions", json={"policy": "human", "session_seed": 1})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["policy"] == "human"

    data = evaluate(sid, 0.06)
    assert data["n_steps"] == 1
    data = evaluate(sid, 0.14)
    assert data["n_steps"] == 2
