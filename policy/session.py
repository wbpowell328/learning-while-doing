"""
Session: the top-level coordinator for one learning-while-doing run.

Each step():
  1. Asks the acquisition policy for the next θ to evaluate.
  2. Runs a deterministic simulation at that θ.
  3. Updates the belief model with the observed total cost.

Common Random Numbers (CRN): step i always calls
    simulate(session_seed, experiment_index=i)
regardless of which c_star was proposed.  This means cost differences across
steps reflect only c_star changes, not different noise draws — a variance
reduction technique that makes the posterior more informative.

Reproducibility: same (session_seed, policy, all configs) → byte-identical run.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from sim import SimConfig, simulate, SimResult
from .belief import BeliefConfig, BeliefModel
from .acquire import AcquisitionConfig, AcquisitionPolicy


@dataclass(frozen=True)
class SessionConfig:
    horizon_weeks: int = 26    # length of each simulated experiment
    best_c_star_grid: int = 200  # grid resolution for best_c_star()


class Session:
    """
    Coordinates a learning-while-doing run.

    Typical usage::

        session = Session(sim_cfg, belief_cfg, acq_cfg, ses_cfg, policy, seed)
        for _ in range(n_experiments):
            result = session.step()
        optimal_c = session.best_c_star()
    """

    def __init__(
        self,
        sim_config: SimConfig,
        belief_config: BeliefConfig,
        acq_config: AcquisitionConfig,
        session_config: SessionConfig,
        policy: AcquisitionPolicy,
        session_seed: int,
    ) -> None:
        self._sim_config = sim_config
        self._acq_config = acq_config
        self._sc = session_config
        self._policy = policy
        self._session_seed = session_seed

        # Acquisition RNG is seeded from session_seed; advances one draw per step()
        self._rng = np.random.default_rng(session_seed)

        self._belief = BeliefModel(belief_config)
        self._history: list[tuple[float, float]] = []   # (c_star, cost)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def step(self) -> SimResult:
        """
        Propose the next θ, run the simulation, update the belief model.

        Returns the full SimResult so callers can inspect daily series, jump
        logs, and cost breakdowns.  Increments n_steps.
        """
        c_star = self._policy.propose(self._belief, self._rng)
        result = simulate(
            config=self._sim_config,
            c_star=c_star,
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
        )
        self._belief.update(c_star, result.total_cost)
        self._history.append((c_star, result.total_cost))
        self._step_count += 1
        return result

    def evaluate(self, c_star: float) -> SimResult:
        """
        Run the simulator at a caller-specified θ, update the belief model.

        Identical CRN contract to step(): uses experiment_index=_step_count so
        the noise path is the same regardless of whether the caller or a policy
        chose the θ value.  Increments n_steps.
        """
        result = simulate(
            config=self._sim_config,
            c_star=float(c_star),
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
        )
        self._belief.update(float(c_star), result.total_cost)
        self._history.append((float(c_star), result.total_cost))
        self._step_count += 1
        return result

    def observe(self, c_star: float, total_cost: float) -> None:
        """
        Inject an external observation (e.g. from a real fund run) into the
        belief model without running a new simulation.

        Does NOT increment n_steps; simulation experiment indices are
        therefore unaffected by observe() calls.
        """
        self._belief.update(float(c_star), float(total_cost))
        self._history.append((float(c_star), float(total_cost)))

    def best_c_star(self) -> float:
        """
        Current best estimate of the optimal θ: argmin of posterior mean.

        With no observations returns the domain midpoint.
        """
        if self._belief.n_observations == 0:
            cfg = self._acq_config
            return (cfg.c_star_min + cfg.c_star_max) / 2.0

        cfg = self._acq_config
        grid = np.linspace(cfg.c_star_min, cfg.c_star_max, self._sc.best_c_star_grid)
        mean, _ = self._belief.posterior(grid)
        return float(grid[np.argmin(mean)])

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
    def history(self) -> list[tuple[float, float]]:
        """Ordered (c_star, cost) pairs from both step() and observe() calls."""
        return list(self._history)
