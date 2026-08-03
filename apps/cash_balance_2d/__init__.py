"""
Cash-balance application — 2-parameter variant (scaffold).

This module is a byte-identical copy of apps/cash_balance/ as a starting
point for a two-parameter (θ = (θ₁, θ₂)) version of the same problem.

TODO for the 2D version — decide what the second parameter is and thread it
through the code:
  * config.py     : add the new field(s) (or leave the shape and let the
                    administrator pass θ as a 2-vector into simulate()).
  * simulator.py  : change simulate()'s signature to accept θ as a
                    numpy array of shape (2,) instead of scalar impparam.
  * dynamics.py   : use both components of θ wherever the current code
                    uses `impparam` (e.g. one might control the target
                    cash ratio and the other the rebalance speed, or one
                    might be a distinct band width around a target).
  * costs.py      : usually unchanged, but revisit if the second parameter
                    affects the cost accounting directly.

Once the 2D version is functionally distinct, register it in
apps/__init__.py (already done under the key "cash_balance_2d") and the
administrator/UI will be able to select it.
"""
from .simulator import simulate
from .config import SimConfig
from .result import SimResult, JumpEvent

# Registry metadata — see apps/base.py Simulator Protocol
NAME       = "cash_balance_2d"
LABEL      = "Cash balance (2-parameter, scaffold)"
MINIMIZE   = True
THETA_DIM  = 2   # will become meaningful once the second parameter is wired in

__all__ = ["simulate", "SimConfig", "SimResult", "JumpEvent",
           "NAME", "LABEL", "MINIMIZE", "THETA_DIM"]
