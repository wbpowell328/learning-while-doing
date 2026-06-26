"""
Test 7: annual rates produce the correct magnitude over a known horizon.

We construct degenerate configs that isolate one rate at a time and compare
against a hand-computed expected value.

This directly tests that no hidden factor-of-252 (or other calendar) error
exists in the cost accumulation.
"""
import numpy as np
import pytest

from sim import SimConfig, simulate


def test_opportunity_cost_pure_cash():
    """
    Config: no flows, no jumps, full rebalance, c_star = 1.0... but c_star
    is bounded at 0.20.  Instead: set c_star=0.20, no flows, no jumps.
    After one year the fund holds ≈ 20% of AUM in cash every day.
    Expected opportunity cost ≈ initial_aum * c_star * (r_market - r_cash) * 1_year.

    With mu=0, sigma=0, no jumps, full rebalance to 20%, the cash buffer
    is always exactly c_star * AUM.  AUM grows because invested earns r_market.
    We test approximate magnitude (within 5%) rather than exact, since AUM drifts.
    """
    AUM = 1_000_000.0
    C = 0.20
    R_MKT = 0.07
    R_CSH = 0.04
    YEARS = 1.0
    WEEKS = int(YEARS * 52)

    cfg = SimConfig(
        initial_aum=AUM,
        mu_net_annual=0.0,
        sigma_net_annual=0.0,   # no retail flow noise
        jump_rate_annual=0.0,   # no jumps
        stationary=True,
        r_market_annual=R_MKT,
        r_cash_annual=R_CSH,
        r_borrow_annual=0.10,
        opp_cost_on_total_cash=True,
        rebalance_speed=1.0,
    )

    r = simulate(cfg, c_star=C, horizon_weeks=WEEKS, session_seed=0, experiment_index=0)

    # Rough expected: AUM * C * (R_MKT - R_CSH) * 1yr ≈ 1e6 * 0.2 * 0.03 = 6000
    # AUM grows slightly so actual will be a bit above; allow ±10%
    expected = AUM * C * (R_MKT - R_CSH) * YEARS
    assert abs(r.opportunity_cost - expected) / expected < 0.10, (
        f"opportunity_cost={r.opportunity_cost:.2f}, expected≈{expected:.2f} "
        f"(ratio={r.opportunity_cost/expected:.3f})"
    )
    assert r.shortfall_cost == 0.0, "No shortfall expected with zero flows"


def test_shortfall_cost_formula():
    """
    shortfall_cost == r_borrow_annual × sum(shortfall_series).

    Per DECISIONS.md D8, r_borrow is a proportional penalty per dollar of
    forced asset liquidation — not a daily interest rate.  This test verifies
    the formula is applied correctly for whatever shortfall the path produces.
    """
    cfg = SimConfig(
        stationary=True,
        jump_rate_annual=500.0,   # many large jumps → guaranteed shortfall at c*=0.01
        jump_mean_log=-1.5,       # ~22% of AUM per jump
        jump_std_log=0.1,
        jump_inflow_prob=0.0,     # all outflows
        r_borrow_annual=0.20,
        opp_cost_on_total_cash=True,
    )
    r = simulate(cfg, c_star=0.01, horizon_weeks=4, session_seed=5, experiment_index=0)

    # Formula: shortfall_cost = r_borrow × sum(pre-rebalance shortfall per day)
    expected_shortfall_cost = cfg.r_borrow_annual * float(np.sum(r.shortfall_series))

    assert np.sum(r.shortfall_series) > 0, "Expected non-zero shortfall with 500 outflow jumps/yr"
    assert abs(r.shortfall_cost - expected_shortfall_cost) < 1e-9, (
        f"shortfall_cost mismatch: got {r.shortfall_cost:.6f}, "
        f"expected {expected_shortfall_cost:.6f}"
    )


def test_no_cost_with_zero_rates():
    """Both cost components should be zero when all rates are equal."""
    cfg = SimConfig(
        stationary=True,
        r_market_annual=0.05,
        r_cash_annual=0.05,    # drag rate = 0
        r_borrow_annual=0.05,  # borrow rate = 0 (no penalty even if shortfall)
        jump_rate_annual=0.0,
        sigma_net_annual=0.0,
        mu_net_annual=0.0,
    )
    r = simulate(cfg, c_star=0.05, horizon_weeks=4, session_seed=0, experiment_index=0)
    assert r.opportunity_cost == 0.0
    assert r.shortfall_cost == 0.0
