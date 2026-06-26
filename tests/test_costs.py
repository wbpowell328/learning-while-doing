"""Test 5: cost identity — total = opportunity + shortfall; both ≥ 0."""
import pytest

from sim import SimConfig, simulate


CFG = SimConfig(stationary=True)


@pytest.mark.parametrize("c_star", [0.01, 0.05, 0.10, 0.20])
@pytest.mark.parametrize("exp_idx", [0, 1, 5])
def test_cost_identity(c_star, exp_idx):
    r = simulate(CFG, c_star=c_star, horizon_weeks=26, session_seed=99, experiment_index=exp_idx)
    assert r.opportunity_cost >= 0.0, f"opportunity_cost={r.opportunity_cost} < 0"
    assert r.shortfall_cost >= 0.0,   f"shortfall_cost={r.shortfall_cost} < 0"
    assert abs(r.total_cost - (r.opportunity_cost + r.shortfall_cost)) < 1e-9, (
        f"total={r.total_cost} ≠ opp={r.opportunity_cost} + shortfall={r.shortfall_cost}"
    )
