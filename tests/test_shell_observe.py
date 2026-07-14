"""
POST /sessions/{sid}/observe tests.

Properties verified:
  1. Returns 200 with n_observations and best_c_star.
  2. n_observations increments with each observe call.
  3. Does NOT increment n_steps (simulate is never called).
  4. best_c_star is in domain.
  5. best_c_star tracks a parabola minimum after enough observations.
  6. Unknown session returns 404.
  7. Missing body fields return 422.
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


def observe(sid: str, c_star: float, total_cost: float) -> dict:
    r = client.post(f"/sessions/{sid}/observe",
                    json={"c_star": c_star, "total_cost": total_cost})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Response shape
# ---------------------------------------------------------------------------

def test_observe_response_fields():
    sid = make_session()
    data = observe(sid, 0.10, 3_000.0)
    assert "n_observations" in data
    assert "best_c_star" in data


# ---------------------------------------------------------------------------
# 2. n_observations increments
# ---------------------------------------------------------------------------

def test_observe_n_observations_increments():
    sid = make_session()
    for expected in range(1, 4):
        data = observe(sid, 0.10, 3_000.0)
        assert data["n_observations"] == expected


# ---------------------------------------------------------------------------
# 3. observe does not increment n_steps
# ---------------------------------------------------------------------------

def test_observe_does_not_increment_n_steps():
    sid = make_session()
    observe(sid, 0.10, 3_000.0)
    observe(sid, 0.12, 4_000.0)
    state = client.get(f"/sessions/{sid}/state").json()
    assert state["n_steps"] == 0
    assert state["n_observations"] == 2


# ---------------------------------------------------------------------------
# 4. best_c_star in domain
# ---------------------------------------------------------------------------

def test_observe_best_c_star_in_bounds():
    sid = make_session()
    data = observe(sid, 0.10, 3_000.0)
    assert ACQ_MIN <= data["best_c_star"] <= ACQ_MAX


# ---------------------------------------------------------------------------
# 5. best_c_star tracks parabola minimum
# ---------------------------------------------------------------------------

def test_observe_tracks_minimum():
    sid = make_session()
    for c in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
        cost = 5_000.0 + 200_000.0 * (c - 0.10) ** 2
        observe(sid, c, cost)
    state = client.get(f"/sessions/{sid}/state").json()
    assert abs(state["best_c_star"] - 0.10) < 0.04


# ---------------------------------------------------------------------------
# 6. Unknown session → 404
# ---------------------------------------------------------------------------

def test_observe_unknown_session():
    r = client.post("/sessions/no-such-id/observe",
                    json={"c_star": 0.10, "total_cost": 1000.0})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 7. Missing body → 422
# ---------------------------------------------------------------------------

def test_observe_missing_body():
    sid = make_session()
    r = client.post(f"/sessions/{sid}/observe", json={})
    assert r.status_code == 422
