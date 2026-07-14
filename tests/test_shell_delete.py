"""
DELETE /sessions/{sid} tests.

Properties verified:
  1. Returns 204 No Content on success.
  2. Session is gone after deletion (subsequent GET /state → 404).
  3. step() on a deleted session returns 404.
  4. Deleting a nonexistent session returns 404.
  5. Deleting the same session twice: first succeeds, second is 404.
"""
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)


def make_session() -> str:
    r = client.post("/sessions", json={
        "sim_config": {"stationary": True},
        "session_config": {"horizon_weeks": 4},
        "policy": "random",
        "session_seed": 1,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


# ---------------------------------------------------------------------------
# 1. Returns 204
# ---------------------------------------------------------------------------

def test_delete_returns_204():
    sid = make_session()
    r = client.delete(f"/sessions/{sid}")
    assert r.status_code == 204


# ---------------------------------------------------------------------------
# 2. State returns 404 after deletion
# ---------------------------------------------------------------------------

def test_delete_removes_state():
    sid = make_session()
    client.delete(f"/sessions/{sid}")
    r = client.get(f"/sessions/{sid}/state")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. step() returns 404 after deletion
# ---------------------------------------------------------------------------

def test_delete_removes_step():
    sid = make_session()
    client.delete(f"/sessions/{sid}")
    r = client.post(f"/sessions/{sid}/step")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 4. Deleting nonexistent session → 404
# ---------------------------------------------------------------------------

def test_delete_nonexistent():
    r = client.delete("/sessions/no-such-session")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. Double-delete: first OK, second 404
# ---------------------------------------------------------------------------

def test_delete_twice():
    sid = make_session()
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.delete(f"/sessions/{sid}").status_code == 404
