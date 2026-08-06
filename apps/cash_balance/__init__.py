"""
Cash-balance application (single-parameter).

Simulates a mutual-fund cash reserve under retail flows, institutional
jumps, and an optional two-state regime. The single implementation
parameter θ (`impparam`) is the target cash-buffer ratio.

See apps/base.py for the Simulator Protocol this module satisfies.
"""
from .simulator import simulate
from .config import SimConfig
from .result import SimResult, JumpEvent
from .dynamics import sample_flows

# Registry metadata — see apps/base.py Simulator Protocol
NAME       = "cash_balance"
LABEL      = "Cash balance (1-parameter)"
MINIMIZE   = False   # maximise expected total reward
THETA_DIM  = 1

__all__ = ["simulate", "sample_flows", "SimConfig", "SimResult", "JumpEvent",
           "NAME", "LABEL", "MINIMIZE", "THETA_DIM"]
