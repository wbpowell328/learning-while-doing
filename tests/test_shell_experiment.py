"""
POST /sessions/{sid}/experiment — the new "control bar" endpoint.

Each call resets the session's belief + history, then executes K+1
iterations: iteration 1 at the user-supplied theta_init, iterations
2..K+1 at policy-picked θ. n_days is the horizon for every iteration.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app, _sessions

client = TestClient(app)


def make_session(policy: str = "random", **kwargs) -> str:
    body = {"policy": policy, "session_seed": 42, **kwargs}
    r = client.post("/sessions", json=body)
    assert r.status_code == 201
    return r.json()["session_id"]


def test_experiment_iteration_count_and_theta_init():
    sid = make_session()
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 50, "policy": "kg", "K": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # 1 initial + 3 policy iterations = 4 total steps.
    assert body["n_steps"] == 4
    # First observation uses the user-supplied θ.
    assert body["history"][0][0] == pytest.approx(0.05)
    # Later observations were policy-picked (non-empty history).
    assert len(body["history"]) == 4


def test_experiment_resets_prior_history():
    """Second Run wipes the first Run's history — reset semantics."""
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment",
                json={"theta_init": 0.05, "n_days": 50, "policy": "kg", "K": 4})
    assert _sessions[sid].n_steps == 5

    r2 = client.post(f"/sessions/{sid}/experiment",
                     json={"theta_init": 0.15, "n_days": 50, "policy": "kg", "K": 1})
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["n_steps"] == 2
    assert b2["history"][0][0] == pytest.approx(0.15)


def test_experiment_k_zero_stops_after_iteration_one():
    """K=0 means one iteration only — the Human-mode path."""
    sid = make_session(policy="human")
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.10, "n_days": 30, "policy": "human", "K": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_steps"] == 1
    assert body["history"][0][0] == pytest.approx(0.10)


def test_experiment_2d_theta_list():
    sid = make_session(app_name="cash_balance_2d")
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": [0.05, 0.15], "n_days": 40, "policy": "ie", "K": 2, "z_alpha": 1.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_steps"] == 3
    # First row is the user-picked 2-vector.
    assert body["history"][0][0] == pytest.approx([0.05, 0.15])


def test_experiment_m_star_takes_effect_on_kg_policy():
    """The m_star sent in the request lands on the KG policy."""
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 20, "policy": "okg", "K": 2, "m_star": 7,
    })
    sess = _sessions[sid]
    assert sess._policy._m_star == 7


def test_experiment_z_alpha_takes_effect_on_ie_policy():
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 20, "policy": "ie", "K": 2, "z_alpha": 2.5,
    })
    sess = _sessions[sid]
    assert sess._acq_config.z_alpha == pytest.approx(2.5)


def test_experiment_rejects_theta_out_of_box():
    """
    Regression: empty θ in the UI used to become Number("") = 0 and hit
    the endpoint, then the sim raised ValueError on impparam < 0.01 and
    bubbled up as a raw 500. Now the endpoint returns 400 with a
    readable message before touching the sim.
    """
    sid = make_session()
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.0, "n_days": 50, "policy": "random", "K": 0,
    })
    assert r.status_code == 400
    assert "outside" in r.json()["detail"]


def test_experiment_rejects_theta_wrong_dim():
    sid = make_session(app_name="cash_balance_2d")
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 50, "policy": "random", "K": 0,   # scalar for 2-D
    })
    assert r.status_code == 400
    assert "dim" in r.json()["detail"]


def test_experiment_rejects_n_days_less_than_one():
    sid = make_session()
    r = client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 0, "policy": "random", "K": 0,
    })
    assert r.status_code == 400


def test_one_more_appends_without_reset():
    """/one_more advances by exactly one step from the current state."""
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 2, "m_star": 3,
    })
    assert _sessions[sid].n_steps == 3

    r = client.post(f"/sessions/{sid}/one_more", json={"n_days": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["n_steps"] == 4
    # First history row is still the original user-picked θ (no reset).
    assert body["history"][0][0] == pytest.approx(0.05)


def test_one_more_can_switch_policy_mid_stream():
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment", json={
        "theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 1, "m_star": 3,
    })
    r = client.post(f"/sessions/{sid}/one_more", json={
        "n_days": 20, "policy": "ie", "z_alpha": 2.5,
    })
    assert r.status_code == 200
    sess = _sessions[sid]
    # Policy swapped, history preserved.
    assert sess._policy.__class__.__name__ == "IEPolicy"
    assert sess._acq_config.z_alpha == pytest.approx(2.5)
    assert sess.n_steps == 3


def test_reset_clears_history_and_does_not_run():
    """
    /reset wipes prior state to initial conditions without executing
    any iteration. The Restart button in the UI wires to this.
    """
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment",
                json={"theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 2})
    assert _sessions[sid].n_steps == 3

    r = client.post(f"/sessions/{sid}/reset")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_steps"] == 0
    assert body["history"] == []
    # Backend state is truly empty — no observations recorded.
    assert _sessions[sid].n_steps == 0
    assert list(_sessions[sid].history) == []


def test_one_more_with_k_runs_k_plus_one_iterations():
    """
    /one_more with K=4 executes 5 more policy-driven iterations
    from the current state (no reset). Symmetric with /experiment.
    """
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment",
                json={"theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 2})
    assert _sessions[sid].n_steps == 3

    r = client.post(f"/sessions/{sid}/one_more",
                    json={"n_days": 20, "K": 4})
    assert r.status_code == 200
    # 3 prior + 5 new = 8.
    assert _sessions[sid].n_steps == 8
    # First row preserved — no reset.
    body = r.json()
    assert body["history"][0][0] == pytest.approx(0.05)


def test_one_more_rejects_bad_n_days():
    sid = make_session()
    client.post(f"/sessions/{sid}/experiment",
                json={"theta_init": 0.05, "n_days": 20, "policy": "kg", "K": 1})
    r = client.post(f"/sessions/{sid}/one_more", json={"n_days": 0})
    assert r.status_code == 400
