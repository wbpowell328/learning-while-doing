"""
Test 6: accounting reconciliation.

cash[t] + invested[t] must equal initial_aum + cumulative_flows[t]
+ cumulative_investment_returns[t].

Because invested assets earn r_market each day, we track the return
component separately and verify money is neither created nor destroyed.
"""
import numpy as np
import pytest

from sim import SimConfig, simulate
from sim.dynamics import run_path


CFG_NO_JUMPS = SimConfig(
    stationary=True,
    jump_rate_annual=0.0,   # zero jumps → pure retail flow, easiest to audit
)

CFG_WITH_JUMPS = SimConfig(stationary=True, jump_rate_annual=12.0)


def _reconcile(config, impparam, horizon_weeks, session_seed, experiment_index):
    """
    Re-run the path and verify the balance sheet at every day.

    AUM[t] = initial_aum + sum(flows[0..t]) + investment_returns[0..t]

    We reconstruct investment_returns by noting:
      invested[t] = invested[t-1] * (1 + r_daily) - transfer[t]
    The transfer[t] is (cash[t] - cash[t-1] - flow[t]).
    So we don't need to expose transfer explicitly — the identity holds at AUM level:
      aum[t] = aum[t-1] + flow[t] + invested[t-1] * r_daily
    """
    cfg = config
    r = simulate(cfg, impparam, horizon_weeks, session_seed, experiment_index)
    dt = 1.0 / cfg.trading_days_per_year
    r_daily = cfg.r_market_annual * dt
    n = r.days

    aum = r.cash_series + r.invested_series
    expected_aum = np.empty(n)
    prev_invested = cfg.initial_aum * (1 - impparam)  # initial invested

    current_aum = cfg.initial_aum
    for t in range(n):
        investment_return = prev_invested * r_daily
        current_aum = current_aum + r.flow_series[t] + investment_return
        expected_aum[t] = current_aum
        # prev_invested for next step = invested at END of this day
        prev_invested = r.invested_series[t]

    np.testing.assert_allclose(
        aum, expected_aum, rtol=1e-10,
        err_msg="cash + invested does not reconcile against cumulative flows + returns"
    )


def test_accounting_no_jumps():
    _reconcile(CFG_NO_JUMPS, impparam=0.05, horizon_weeks=26,
               session_seed=1, experiment_index=0)


def test_accounting_with_jumps():
    _reconcile(CFG_WITH_JUMPS, impparam=0.05, horizon_weeks=26,
               session_seed=1, experiment_index=0)


def test_accounting_low_impparam():
    _reconcile(CFG_WITH_JUMPS, impparam=0.01, horizon_weeks=26,
               session_seed=2, experiment_index=3)


def test_accounting_high_impparam():
    _reconcile(CFG_WITH_JUMPS, impparam=0.20, horizon_weeks=26,
               session_seed=3, experiment_index=1)
