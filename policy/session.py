"""
Session: the top-level coordinator for one learning-while-doing run.

Each step():
  1. Asks the acquisition policy for the next θ to evaluate.
  2. Runs a deterministic simulation at that θ.
  3. Updates the belief model with the observed total cost.

θ may be a scalar (1-D problem) or a vector (multi-dim problem).  The
session dispatches to the correct application module and threads θ
through the sim, belief, and history plumbing.

Common Random Numbers (CRN): step i always calls simulate() with
`experiment_index=i` regardless of which θ was proposed.  This means cost
differences across steps reflect only θ changes, not different noise
draws — a variance-reduction technique that makes the posterior more
informative.

Reproducibility: same (session_seed, policy, all configs) → byte-identical run.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

from apps.cash_balance import SimConfig as CashBalanceSimConfig, simulate as cash_balance_simulate
from .belief import BeliefConfig, BeliefModel
from .acquire import AcquisitionConfig, AcquisitionPolicy, _make_grid


@dataclass(frozen=True)
class SessionConfig:
    horizon_weeks: int = 26         # length of each simulated experiment
    best_impparam_grid: int = 200   # per-dim grid resolution for best_impparam()


def _dim_of(cfg: AcquisitionConfig) -> int:
    """How many components does θ have, per the acquisition-config bounds?"""
    return int(np.atleast_1d(np.asarray(cfg.impparam_min, dtype=float)).size)


class Session:
    """
    Coordinates a learning-while-doing run.

    Typical usage::

        session = Session(sim_cfg, belief_cfg, acq_cfg, ses_cfg, policy, seed)
        for _ in range(n_experiments):
            result = session.step()
        optimal = session.best_impparam()

    Multi-app dispatch:
        `simulate_fn` (optional) — pluggable simulate function.  Defaults to
        `apps.cash_balance.simulate` so existing callers behave identically.
        Pass e.g. `apps.cash_balance_2d.simulate` for the 2-parameter variant.
    """

    def __init__(
        self,
        sim_config,                     # app-specific SimConfig
        belief_config: BeliefConfig,
        acq_config: AcquisitionConfig,
        session_config: SessionConfig,
        policy: AcquisitionPolicy,
        session_seed: int,
        simulate_fn: Callable | None = None,
    ) -> None:
        self._sim_config = sim_config
        self._acq_config = acq_config
        self._sc = session_config
        self._policy = policy
        self._session_seed = session_seed
        self._simulate = simulate_fn if simulate_fn is not None else cash_balance_simulate

        self._dim = _dim_of(acq_config)

        # Acquisition RNG is seeded from session_seed; advances one draw per step()
        self._rng = np.random.default_rng(session_seed)

        self._belief = BeliefModel(belief_config, dim=self._dim)
        # For dim=1 history entries look like (float, float) as before;
        # for dim≥2 they are (np.ndarray, float).
        self._history: list[tuple] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def step(self):
        """Propose the next θ, run the simulation, update the belief."""
        impparam = self._policy.propose(self._belief, self._rng)
        result = self._simulate(
            config=self._sim_config,
            impparam=impparam,
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
        )
        self._belief.update(impparam, result.total_cost)
        self._history.append((impparam, result.total_cost))
        self._step_count += 1
        return result

    def evaluate(self, impparam):
        """Run the simulator at a caller-specified θ (scalar or vector)."""
        result = self._simulate(
            config=self._sim_config,
            impparam=impparam,
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
        )
        self._belief.update(impparam, result.total_cost)
        self._history.append((impparam, result.total_cost))
        self._step_count += 1
        return result

    def observe(self, impparam, total_cost: float) -> None:
        """Inject an external observation without running a new simulation."""
        self._belief.update(impparam, float(total_cost))
        self._history.append((impparam, float(total_cost)))

    def best_impparam(self):
        """
        Current best estimate of the optimal θ: argmin of posterior mean over
        a fine grid.  Returns scalar for dim=1, ndarray for dim≥2.  With no
        observations returns the box midpoint.
        """
        cfg = self._acq_config
        lo = np.atleast_1d(np.asarray(cfg.impparam_min, dtype=float))
        hi = np.atleast_1d(np.asarray(cfg.impparam_max, dtype=float))

        if self._belief.n_observations == 0:
            mid = 0.5 * (lo + hi)
            return float(mid[0]) if self._dim == 1 else mid

        grid = _make_grid(cfg.impparam_min, cfg.impparam_max, self._sc.best_impparam_grid)
        mean, _ = self._belief.posterior(grid)
        idx = int(np.argmin(mean))
        best = grid[idx]
        return float(best[0]) if self._dim == 1 else best.copy()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Number of simulate() calls made (does not count observe())."""
        return self._step_count

    @property
    def belief(self) -> BeliefModel:
        return self._belief

    @property
    def history(self) -> list:
        """Ordered (impparam, cost) pairs from step() and observe() calls."""
        return list(self._history)

    @property
    def dim(self) -> int:
        return self._dim
