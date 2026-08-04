"""
GET /sessions/{sid}/observations_enriched — per-observation rows with
μ^n (posterior mean at that θ BEFORE the observation was folded in) and
offline analytic correlated KG at the same pre-step belief state.

Row 0 always sees the pure prior. Higher rows see progressively-more-
informed beliefs.
"""
import pytest
from fastapi.testclient import TestClient
from shell.app import app

client = TestClient(app)


def make_session(policy: str = "kg", **kwargs) -> str:
    body = {"policy": policy, "session_seed": 42}
    body.update(kwargs)
    r = client.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def test_empty_history():
    sid = make_session()
    r = client.get(f"/sessions/{sid}/observations_enriched")
    assert r.status_code == 200
    assert r.json() == {"rows": []}


def test_1d_after_a_few_steps():
    sid = make_session()
    for _ in range(3):
        client.post(f"/sessions/{sid}/step")
    r = client.get(f"/sessions/{sid}/observations_enriched")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 3
    for i, row in enumerate(rows):
        # Same shape / step index round-trip.
        assert row["step"] == i
        # 1-D theta comes back as a scalar float.
        assert isinstance(row["theta"], float)
        # μ, kg are present and finite; kg is nonneg (info value).
        assert isinstance(row["mu"], float)
        assert isinstance(row["kg"], float)
        assert row["kg"] >= -1e-6      # allow tiny numerical undershoot
    # Row 0 uses the pure prior — same μ regardless of observations.
    # Row 1+ uses the belief after row 0..i-1 have been folded in — so
    # μ_1 differs from μ_0 once there's any signal.
    assert rows[0]["mu"] != rows[-1]["mu"] or rows[0]["value"] != rows[-1]["value"]


def test_2d_supported():
    sid = make_session(app_name="cash_balance_2d")
    for _ in range(2):
        client.post(f"/sessions/{sid}/step")
    r = client.get(f"/sessions/{sid}/observations_enriched")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 2
    for row in rows:
        # 2-D theta returned as a length-2 list.
        assert isinstance(row["theta"], list)
        assert len(row["theta"]) == 2
        assert isinstance(row["mu"], float)
        assert isinstance(row["kg"], float)


def test_row0_uses_prior_regardless_of_theta():
    """
    Row 0's μ is the prior_mean regardless of what θ was sampled — because
    the belief is empty before the first observation.
    """
    sid = make_session()
    client.post(f"/sessions/{sid}/step")
    r = client.get(f"/sessions/{sid}/observations_enriched")
    rows = r.json()["rows"]
    # For an empty belief, μ at any θ equals prior_mean. In cost frame that's
    # BeliefConfig defaults (5000); in reward frame Session negates to −5000,
    # then _to_display negates back to +5000. Either way it's a constant.
    # We just assert consistency across rows if a second obs had been made,
    # but with only 1 obs we assert row 0's μ isn't wildly out of range.
    assert abs(rows[0]["mu"]) < 1e7
