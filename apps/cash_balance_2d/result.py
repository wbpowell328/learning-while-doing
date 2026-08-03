from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class JumpEvent(NamedTuple):
    day: int
    size_fraction: float  # |jump| / aum_inst at jump time; always positive
    direction: int        # +1 = inflow, -1 = outflow


@dataclass
class SimResult:
    # --- Identity: enough to replay byte-identically ---
    impparam: np.ndarray        # shape (2,): (theta_ind, theta_inst)
    session_seed: int
    experiment_index: int

    # --- Reward breakdown (user-facing; app is a MAXIMISATION problem) ---
    total_reward: float               # = market_gain + cash_gain - shortfall_ind_penalty - shortfall_inst_penalty
    market_gain: float
    cash_gain: float
    shortfall_ind_penalty: float
    shortfall_inst_penalty: float

    # --- Legacy cost breakdown (internal; belief layer keeps minimising) ---
    total_cost: float                # = opportunity_cost + shortfall_ind_cost + shortfall_inst_cost
    opportunity_cost: float          # cumulative cash drag over horizon
    shortfall_ind_cost: float        # cumulative deferred-payment penalty (individual)
    shortfall_inst_cost: float       # cumulative forced-liquidation penalty (institutional)

    @property
    def shortfall_cost(self) -> float:
        """Convenience alias — total shortfall cost across both investor classes."""
        return self.shortfall_ind_cost + self.shortfall_inst_cost

    @property
    def shortfall_penalty(self) -> float:
        """Convenience alias — total shortfall reward-penalty (both classes)."""
        return self.shortfall_ind_penalty + self.shortfall_inst_penalty

    # --- Daily time series (length = trading days in horizon) ---
    cash_series: np.ndarray          # cash balance end-of-day (post-rebalance)
    invested_series: np.ndarray      # invested-asset value end-of-day (post-rebalance)
    aum_ind_series: np.ndarray       # individual investors' AUM end-of-day
    aum_inst_series: np.ndarray      # institutional investors' AUM end-of-day
    flow_ind_series: np.ndarray      # net daily individual flow
    flow_inst_series: np.ndarray     # net daily institutional flow
    shortfall_ind_series: np.ndarray # daily deferred-individual shortfall (basis for ind cost)
    shortfall_inst_series: np.ndarray# daily immediate-institutional shortfall (basis for inst cost)

    # --- Institutional jump log ---
    event_log: list[JumpEvent]

    # --- Metadata ---
    days: int  # length of all series
