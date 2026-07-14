"""
Acquisition policies for the learning-while-doing optimizer.

Each policy implements one method:
    propose(model: BeliefModel, rng: np.random.Generator) -> float

returning the next C* to evaluate.  Policies are stateless given
(model, rng); all randomness flows through the provided rng.

Policies implemented here (in order of complexity):
  RandomPolicy — uniform random baseline; ignores the model entirely.
  IEPolicy     — information-theoretic exploration: argmax posterior std.
  KGPolicy     — knowledge gradient: argmax expected improvement in best mean.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .belief import BeliefModel


from typing import Protocol


class AcquisitionPolicy(Protocol):
    """Structural interface satisfied by RandomPolicy, IEPolicy, and KGPolicy."""
    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float: ...


@dataclass(frozen=True)
class AcquisitionConfig:
    c_star_min: float = 0.01
    c_star_max: float = 0.20
    grid_size: int = 100   # evaluation grid resolution for IE and KG
    kg_n_mc: int = 500     # Monte Carlo samples for KG expectation


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomPolicy:
    """
    Proposes a C* drawn uniformly at random from [c_star_min, c_star_max].
    The BeliefModel is not consulted; the rng drives all randomness.
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        cfg = self.config
        return float(rng.uniform(cfg.c_star_min, cfg.c_star_max))


# ---------------------------------------------------------------------------
# Information-theoretic exploration (IE)
# ---------------------------------------------------------------------------

class IEPolicy:
    """
    Proposes the C* with the highest posterior standard deviation on the grid.

    This is the one-step maximum entropy reduction acquisition under a GP with
    independent observations: the point that, when observed, maximally reduces
    uncertainty about F.  Pure exploration — does not exploit the posterior mean.

    With no observations the prior std is uniform, so the first proposal is
    always grid[0] (= c_star_min).  Subsequent proposals move to the region
    of highest remaining uncertainty.
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        _, std = model.posterior(self._grid)
        return float(self._grid[np.argmax(std)])


# ---------------------------------------------------------------------------
# Knowledge Gradient (KG)
# ---------------------------------------------------------------------------

class KGPolicy:
    """
    Proposes the C* that maximally improves the expected posterior-mean minimum.

    KG(x) = min_j μ_n(grid_j) − E_z[ min_j (μ_n(grid_j) + w_j(x) · z) ]

    where z ~ N(0,1) and the influence vector is
        w_j(x) = Cov_n(x, grid_j) / √(Var_n(x) + σ²_noise)

    KG ≥ 0 always (Jensen's inequality); Monte Carlo noise can produce small
    negative estimates which are harmless for argmax selection.

    The expectation is estimated by Monte Carlo using the provided rng, so
    proposals are reproducible given the same rng state.
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        kg = self._kg_values(model, rng)
        return float(self._grid[np.argmax(kg)])

    def kg_values(self, model: BeliefModel, rng: np.random.Generator) -> np.ndarray:
        """Return KG for every grid point (length = grid_size). Useful for diagnostics."""
        return self._kg_values(model, rng)

    def _kg_values(self, model: BeliefModel, rng: np.random.Generator) -> np.ndarray:
        grid = self._grid
        cfg = self.config

        mu, _ = model.posterior(grid)                          # (m,)
        current_min = float(np.min(mu))

        post_cov = model.posterior_cov_matrix(grid)            # (m, m)
        post_var = np.maximum(np.diag(post_cov), 0.0)         # (m,)
        noise_var = model.config.noise_std ** 2

        sigma_total = np.sqrt(post_var + noise_var)            # (m,)
        # Avoid division by zero; those rows get w = 0 → KG = 0
        safe_sigma = np.where(sigma_total > 1e-12, sigma_total, 1.0)
        W = post_cov / safe_sigma[:, None]                     # (m, m)
        # Zero out rows where sigma_total was negligible
        W[sigma_total <= 1e-12, :] = 0.0

        z = rng.standard_normal(cfg.kg_n_mc)                   # (n_mc,)

        # future_means[i, k] = min_j(mu[j] + W[i,j] * z[k])
        # shape: (m, n_mc, m) → min over last axis → (m, n_mc)
        future_min = np.min(
            mu[None, None, :] + W[:, None, :] * z[None, :, None],
            axis=-1,
        )                                                       # (m, n_mc)

        expected_future_min = future_min.mean(axis=1)          # (m,)
        return current_min - expected_future_min                # (m,)
