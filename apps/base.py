"""
Simulator contract that every pluggable application must satisfy.

Concept: an "application" is a self-contained problem — its stochastic
model, cost accounting, and the domain of the implementation parameter
θ. It knows nothing about learning policies or the administrator that
drives it. Given θ (scalar or vector), a seed, and a step index, the
simulator returns a scalar cost and any application-specific auxiliary
data.

The administrator is responsible for choosing θ (via a policy) and for
tracking the observation history. Reporting is application-agnostic and
consumes only the scalar cost.
"""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Simulator(Protocol):
    """
    Structural interface for a pluggable application.

    Required class-level metadata
    -----------------------------
    NAME       : short registry key, e.g. "cash_balance".
    LABEL      : human-readable name for the UI.
    MINIMIZE   : True if lower cost is better; False for maximization.
    THETA_DIM  : dimension of θ (1 for scalar, 2+ for vector).

    Required methods
    ----------------
    theta_bounds() -> (lower, upper)
        Numpy arrays of shape (THETA_DIM,) giving the legal box for θ.

    simulate(theta, session_seed, experiment_index, **kwargs) -> result
        Runs one full evaluation at θ (typically a multi-period path)
        with the deterministic seed (session_seed, experiment_index).
        Returns an object with at least a `.total_cost` float attribute.
        May carry additional per-application data on the result object
        (e.g. cash_series, event_log for the cash-balance app).
    """
    NAME: str
    LABEL: str
    MINIMIZE: bool
    THETA_DIM: int

    def theta_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    def simulate(
        self,
        theta: float | np.ndarray,
        session_seed: int,
        experiment_index: int,
        **kwargs: Any,
    ) -> Any: ...
