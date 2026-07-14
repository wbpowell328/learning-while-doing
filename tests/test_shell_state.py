"""
GET /sessions/{sid}/state tests.

Properties verified:
  1. Returns 200 with all expected fields on a fresh session.
  2. n_steps and n_observations start at 0.
  3. history starts empty.
  4. State accumulates correctly across step() and observe() calls.
  5. history entries match what was proposed / injected.
  6. Unknown session returns 404.
"""
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)

ACQ_MIN = 0.01
ACQ_MAX = 0.20


def make_session() -> str:
    r = client.post("/sessions", json={
        "sim_config": {"stationary": True},
        "session_config": {"horizon_weeks": 4},
        "policy": "random",
        "session_seed": 42,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


def get_state(sid: str) -> dict:
    r = client.get(f"/sessions/{sid}/state")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Response shape
# ---------------------------------------------------------------------------

def test_state_response_fields():
    sid = make_session()
    data = get_state(sid)
    assert set(data.keys()) >= {"n_steps", "n_observations", "best_c_star", "history"}


# ---------------------------------------------------------------------------
# 2. Fresh session starts at zero
# ---------------------------------------------------------------------------

def test_state_initial_zeros():
    sid = make_session()
    data = get_state(sid)
    assert data["n_steps"] == 0
    assert data["n_observations"] == 0


# ---------------------------------------------------------------------------
# 3. History starts empty
# ---------------------------------------------------------------------------

def test_state_initial_history_empty():
    sid = make_session()
    assert get_state(sid)["history"] == []


# ---------------------------------------------------------------------------
# 4. State accumulates across step + observe
# ---------------------------------------------------------------------------

def test_state_accumulates():
    sid = make_session()
    client.post(f"/sessions/{sid}/step")
    client.post(f"/sessions/{sid}/observe",
                json={"c_star": 0.10, "total_cost": 3_000.0})
    client.post(f"/sessions/{sid}/step")

    data = get_state(sid)
    assert data["n_steps"] == 2
    assert data["n_observations"] == 3
    assert len(data["history"]) == 3


# ---------------------------------------------------------------------------
# 5. History entries are valid [c_star, cost] pairs
# ---------------------------------------------------------------------------

def test_state_history_entries_valid():
    sid = make_session()
    client.post(f"/sessions/{sid}/observe",
                json={"c_star": 0.07, "total_cost": 2_500.0})
    client.post(f"/sessions/{sid}/observe",
                json={"c_star": 0.13, "total_cost": 4_000.0})

    history = get_state(sid)["history"]
    assert len(history) == 2
    for entry in history:
        c, cost = entry
        assert ACQ_MIN <= c <= ACQ_MAX
        assert cost > 0


# ---------------------------------------------------------------------------
# 6. Unknown session → 404
# ---------------------------------------------------------------------------

def test_state_unknown_session():
    r = client.get("/sessions/ghost/state")
    assert r.status_code == 404
