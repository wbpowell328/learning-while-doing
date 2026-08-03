from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class JumpEvent(NamedTuple):
    day: int
    size_fraction: float  # |jump| / AUM at jump time; always positive
    direction: int        # +1 = inflow, -1 = outflow


@dataclass
class SimResult:
    # --- Identity: enough to replay byte-identically ---
    impparam: float
    session_seed: int
    experiment_index: int

    # --- Cost breakdown ---
    total_cost: float        # = opportunity_cost + shortfall_cost
    opportunity_cost: float  # cumulative cash drag over horizon
    shortfall_cost: float    # cumulative borrowing penalty over horizon

    # --- Daily time series (length = trading days in horizon) ---
    cash_series: np.ndarray       # cash balance at end of each day (post-rebalance)
    invested_series: np.ndarray   # invested-asset value at end of each day (post-rebalance)
    flow_series: np.ndarray       # net daily flow (retail + jumps); for accounting
    shortfall_series: np.ndarray  # pre-rebalance shortfall per day; basis for shortfall_cost

    # --- Institutional jump log ---
    event_log: list[JumpEvent]

    # --- Metadata ---
    days: int  # length of all series

    # --- Future belief-model consumer hook ---
    # Populated by a batch helper; None for a single run.
    variance_estimate: float | None = None
