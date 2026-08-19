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
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from apps.cash_balance import SimConfig as CashBalanceSimConfig, simulate as cash_balance_simulate
from .belief import BeliefConfig, BeliefModel
from .acquire import AcquisitionConfig, AcquisitionPolicy, _make_grid


@dataclass(frozen=True)
class SessionConfig:
    horizon_weeks: int = 10         # length of each simulated experiment (50 trading days — matches the ExperimentBar's default N)
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
        minimize: bool = True,
        m_star: int = 1,
        budget: int | None = None,
        kg_mult: float = 1.0,
    ) -> None:
        self._sim_config = sim_config
        self._acq_config = acq_config
        self._sc = session_config
        self._policy = policy
        self._session_seed = session_seed
        self._simulate = simulate_fn if simulate_fn is not None else cash_balance_simulate
        self._minimize = bool(minimize)
        self._m_star = max(1, int(m_star))
        # Ryzhov-style budget N — used by OKGRyzhov*Policy for the (N-n)
        # exploration multiplier. Preserved on the Session so mid-session
        # policy swaps can reconstruct the Ryzhov policy without losing
        # N (create-time is the only place req.budget is available).
        self._budget = budget
        # TEMP research knob M — linear KG multiplier for the online-
        # correlated policy. Kept on the Session so policy swaps preserve it.
        self._kg_mult = max(0.0, float(kg_mult))
        # Push m_star into the policy if it supports it (all KG variants do).
        if hasattr(policy, "set_m_star"):
            policy.set_m_star(self._m_star)
        if hasattr(policy, "set_kg_mult"):
            policy.set_kg_mult(self._kg_mult)

        self._dim = _dim_of(acq_config)

        # Acquisition RNG is seeded from session_seed; advances one draw per step()
        self._rng = np.random.default_rng(session_seed)

        # For maximisation apps, the caller specifies prior_mean in the
        # reward frame (what a typical run pays). The belief layer keeps
        # operating on "value to minimise" (=-reward), so negate the prior
        # mean here. signal_std and noise_std are variances → frame-
        # independent, no change.
        belief_config_internal = belief_config
        if not self._minimize:
            belief_config_internal = replace(
                belief_config, prior_mean=-float(belief_config.prior_mean),
            )
        self._belief = BeliefModel(belief_config_internal, dim=self._dim)
        # For dim=1 history entries look like (float, float) as before;
        # for dim≥2 they are (np.ndarray, float).
        self._history: list[tuple] = []
        # Parallel list: n_days per observation (batch length). Needed
        # so a refit (e.g. length_scale change) can replay the belief
        # in per-day units even when past batches had different
        # lengths. Kept parallel to _history rather than folded into
        # the tuple so existing (θ, value) unpacking keeps working.
        self._history_n_days: list[int] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def _observed_value(self, result, n_days: int) -> float:
        """
        The scalar this run contributes to the belief (per-day frame).

        The belief operates on reward-per-day so mixed batch lengths
        stay coherent — if F(θ) is truly the daily expected reward,
        a 50-day run and a 200-day run at the same θ produce the same
        expected per-day observation. Without this normalization, the
        GP would see 4x-different y-values for the same θ and get
        badly poisoned when a user changes N mid-session (Warren
        noticed this 2026-08).

        Belief always minimises internally, so for a maximisation app
        we feed -reward. History stores the display-frame *batch*
        value so callers keep seeing per-batch dollars.
        """
        n = max(1, int(n_days))
        if self._minimize:
            return float(result.total_cost) / n
        return -float(result.total_reward) / n

    def _display_value(self, result) -> float:
        if self._minimize:
            return float(result.total_cost)
        return float(result.total_reward)

    def _per_obs_noise(self, n_days: int) -> float:
        """
        Noise std for a single per-day-average observation over n_days.
        The belief's config.noise_std is the per-DAY noise; the average
        of n_days iid daily samples has std σ_n / √n_days. Feeding this
        as the observation's noise makes the belief trust longer batches
        more, which is the whole point of accumulating days.
        """
        import math
        n = max(1, int(n_days))
        return float(self._belief.config.noise_std) / math.sqrt(n)

    def _effective_n_days(self, n_days: int | None) -> int:
        """
        Resolve the batch length in trading days.

        If the caller passes n_days, use that. Otherwise default to
        horizon_weeks * 5 (the session's baseline). Never returns < 1
        (the belief-side divide-by-n_days would explode).
        """
        n = int(n_days) if n_days is not None else int(self._sc.horizon_weeks) * 5
        return max(1, n)

    def step(self, n_days: int | None = None):
        """
        Propose the next θ, run the simulation, update the belief.

        `n_days` (optional) overrides the session's default horizon for
        this one iteration only — the /experiment endpoint uses this
        so the user's chosen "N days" applies to every iteration.
        """
        n = self._effective_n_days(n_days)
        impparam = self._policy.propose(self._belief, self._rng)
        result = self._simulate(
            config=self._sim_config,
            impparam=impparam,
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
            n_days=n,
        )
        self._belief.update(impparam, self._observed_value(result, n),
                            noise_std=self._per_obs_noise(n))
        self._history.append((impparam, self._display_value(result)))
        self._history_n_days.append(n)
        self._step_count += 1
        return result

    def evaluate(self, impparam, n_days: int | None = None):
        """Run the simulator at a caller-specified θ (scalar or vector)."""
        n = self._effective_n_days(n_days)
        result = self._simulate(
            config=self._sim_config,
            impparam=impparam,
            horizon_weeks=self._sc.horizon_weeks,
            session_seed=self._session_seed,
            experiment_index=self._step_count,
            n_days=n,
        )
        self._belief.update(impparam, self._observed_value(result, n),
                            noise_std=self._per_obs_noise(n))
        self._history.append((impparam, self._display_value(result)))
        self._history_n_days.append(n)
        self._step_count += 1
        return result

    def observe(self, impparam, total_cost: float, n_days: int | None = None) -> None:
        """
        Inject an external observation without running a new simulation.
        Argument is in the app's display frame (cost for min, reward for max).
        `n_days` defaults to horizon_weeks*5; used to scale the observation
        into the belief's per-day frame.
        """
        v = float(total_cost)
        n = self._effective_n_days(n_days)
        # Belief always minimises; negate reward for maximise apps.
        internal_batch = v if self._minimize else -v
        self._belief.update(impparam, internal_batch / n,
                            noise_std=self._per_obs_noise(n))
        self._history.append((impparam, v))
        self._history_n_days.append(n)

    def best_impparam(self):
        """
        Current best θ. Internally belief always stores "value to minimise",
        so argmin(mean) is correct in both frames (it becomes argmax(reward)
        for maximise apps because the internal value is -reward).

        Returns scalar for dim=1, ndarray for dim≥2.  With no observations
        returns the box midpoint.
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
        """
        Ordered (impparam, value) pairs from step()/observe() calls, where
        `value` is in the app's display frame: total_cost for minimise apps,
        total_reward for maximise apps.
        """
        return list(self._history)

    @property
    def history_n_days(self) -> list[int]:
        """
        Parallel to `history` — the batch length (trading days) each
        observation was collected over. Callers doing per-day↔per-batch
        conversion for display (e.g. the μⁿ column in the observations
        table) need this per-row.
        """
        return list(self._history_n_days)

    @property
    def minimize(self) -> bool:
        return self._minimize

    @property
    def m_star(self) -> int:
        """Batch-size trick multiplier used by KG-family policies."""
        return self._m_star

    @property
    def budget(self) -> int | None:
        """Ryzhov-style N — used by OKGRyzhov policies for (N-n)·KG."""
        return self._budget

    def set_m_star(self, m_star: int) -> None:
        """
        Change m* for the current session's policy. Effective on the NEXT
        policy.propose() call; past steps are not recomputed. No-op if the
        policy doesn't support it (Random / IE / Human).
        """
        self._m_star = max(1, int(m_star))
        if hasattr(self._policy, "set_m_star"):
            self._policy.set_m_star(self._m_star)

    def set_kg_mult(self, kg_mult: float) -> None:
        """TEMP research knob M — linear KG multiplier for the online-
        correlated policy. Effective on the next propose(); no-op for
        policies that don't support it."""
        self._kg_mult = max(0.0, float(kg_mult))
        if hasattr(self._policy, "set_kg_mult"):
            self._policy.set_kg_mult(self._kg_mult)

    @property
    def kg_mult(self) -> float:
        return self._kg_mult

    def reset(self) -> None:
        """
        Clear observations and refit the belief from scratch using the
        current belief config (same length_scale, signal_std, noise_std,
        prior_mean). Also re-seed the acquisition RNG so the same
        experiment configuration reproduces byte-for-byte.

        The /experiment endpoint calls this before every Run so each
        experiment starts from the same prior state.
        """
        self._belief = BeliefModel(self._belief.config, dim=self._dim)
        self._history = []
        self._history_n_days = []
        self._step_count = 0
        self._rng = np.random.default_rng(self._session_seed)

    def set_policy(self, policy) -> None:
        """
        Swap the acquisition policy on a running session. The new policy
        inherits the session's current m*. Used by the /experiment
        endpoint when the user changes policy in the control bar.
        """
        self._policy = policy
        if hasattr(policy, "set_m_star"):
            policy.set_m_star(self._m_star)
        if hasattr(policy, "set_kg_mult"):
            policy.set_kg_mult(self._kg_mult)

    def set_length_scale(self, length_scale) -> None:
        """
        Rebuild the belief with a new kernel length_scale, replaying the
        session's history through the fresh posterior. Same observations,
        different smoothness assumption — the GP posterior line/surface
        will look tighter (small length_scale) or smoother (large).

        Accepts a scalar (isotropic) or a length-`dim` sequence (ARD).
        """
        # Preserve everything on the config except length_scale.
        new_config = replace(self._belief.config, length_scale=length_scale)
        new_belief = BeliefModel(new_config, dim=self._dim)
        # session.history is in display frame (per-batch dollars); the
        # belief operates on per-day values, so divide by that row's
        # n_days when replaying. Negate for maximise apps so the belief
        # sees "value to minimise".
        for (impparam, disp_val), n_days in zip(self._history, self._history_n_days):
            n = max(1, int(n_days))
            per_day = float(disp_val) / n if self._minimize else -float(disp_val) / n
            new_belief.update(impparam, per_day,
                              noise_std=self._per_obs_noise(n))
        self._belief = new_belief

    @property
    def dim(self) -> int:
        return self._dim
