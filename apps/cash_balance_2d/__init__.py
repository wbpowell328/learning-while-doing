"""
Cash-balance application — 2-parameter variant.

Two investor classes tracked separately:
  * Individual   — frequent, small deposits/withdrawals (Brownian on aum_ind).
  * Institutional — rare, large deposits/withdrawals (Poisson × lognormal on aum_inst).

Policy: two-vector θ = (θ_ind, θ_inst).  Target cash at end of day is
  target_cash = θ_ind · aum_ind + θ_inst · aum_inst.

Shortfall attribution when cash < 0 after the day's flows:
  * Institutional-caused portion (min of |cash| and |inst outflow today|)
    triggers immediate liquidation from `invested` and incurs the high
    penalty rate r_borrow_inst_annual (default 0.10).
  * Individual-caused portion (the remainder) is deferred and incurs the
    low rate r_borrow_ind_annual (default 0.02).

Expected optimum shape: θ_inst* > θ_ind* — you hoard cash against the
expensive institutional shortfall and let the cheap-to-defer individual
one absorb the pressure.  See tests/test_cash_balance_2d.py for a
scan verifying an interior minimum near (0.02, 0.26) with default params.
"""
from .simulator import simulate
from .config import SimConfig
from .result import SimResult, JumpEvent

# Registry metadata — see apps/base.py Simulator Protocol
NAME       = "cash_balance_2d"
LABEL      = "Cash balance (2-parameter, scaffold)"
MINIMIZE   = False   # maximise expected total reward
THETA_DIM  = 2   # will become meaningful once the second parameter is wired in

__all__ = ["simulate", "SimConfig", "SimResult", "JumpEvent",
           "NAME", "LABEL", "MINIMIZE", "THETA_DIM"]
