"""
POST /sessions tests.

Properties verified:
  1. Default request returns 201 and a session_id string.
  2. Response echoes back the requested policy.
  3. Two calls produce different session IDs.
  4. Works with each of the three policy names.
  5. Invalid policy name returns 422.
  6. Custom sim_config fields are accepted without error.
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)


def create_default() -> dict:
    r = client.post("/sessions", json={})
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Returns 201 + session_id
# ---------------------------------------------------------------------------

def test_create_returns_201():
    r = client.post("/sessions", json={})
    assert r.status_code == 201


def test_create_returns_session_id():
    data = create_default()
    assert "session_id" in data
    # Must be a valid UUID
    uuid.UUID(data["session_id"])


# ---------------------------------------------------------------------------
# 2. Response echoes policy
# ---------------------------------------------------------------------------

def test_create_echoes_policy():
    r = client.post("/sessions", json={"policy": "ie"})
    assert r.status_code == 201
    assert r.json()["policy"] == "ie"


# ---------------------------------------------------------------------------
# 3. Different session IDs per call
# ---------------------------------------------------------------------------

def test_create_unique_ids():
    ids = {create_default()["session_id"] for _ in range(3)}
    assert len(ids) == 3, "Expected unique session IDs each call"


# ---------------------------------------------------------------------------
# 4. All three policy names accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["random", "ie", "kg"])
def test_create_all_policies(policy):
    r = client.post("/sessions", json={"policy": policy})
    assert r.status_code == 201
    assert r.json()["policy"] == policy


# ---------------------------------------------------------------------------
# 5. Invalid policy returns 422
# ---------------------------------------------------------------------------

def test_create_invalid_policy():
    r = client.post("/sessions", json={"policy": "banana"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 6. Custom config fields accepted
# ---------------------------------------------------------------------------

def test_create_custom_sim_config():
    r = client.post("/sessions", json={
        "sim_config": {"stationary": True, "jump_rate_annual": 6.0},
        "session_config": {"horizon_weeks": 4},
        "policy": "random",
        "session_seed": 99,
    })
    assert r.status_code == 201
