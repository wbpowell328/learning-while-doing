"""
Acquisition policies for the learning-while-doing optimizer.

Each policy implements one method:
    propose(model: BeliefModel, rng: np.random.Generator) -> float

returning the next C* to evaluate.  Policies are stateless given
(model, rng); all randomness flows through the provided rng.

Policies implemented here (in order of complexity):
  RandomPolicy — uniform random baseline; ignores the model entirely.
  IEPolicy     — information-theoretic exploration: argmax posterior std.
  KGPolicy     — knowledge gradient for correlated beliefs (analytic).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .belief import BeliefModel


class AcquisitionPolicy(Protocol):
    """Structural interface satisfied by RandomPolicy, IEPolicy, and KGPolicy."""
    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float: ...


@dataclass(frozen=True)
class AcquisitionConfig:
    c_star_min: float = 0.01
    c_star_max: float = 0.20
    grid_size: int = 100   # evaluation grid resolution for IE and KG


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
# Knowledge Gradient (KG) — correlated beliefs, closed-form
# ---------------------------------------------------------------------------

# Standard-normal helpers (avoid a scipy dependency).
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # math.erf is a C function; loop is fine — z has O(#envelope segments) entries.
    return np.array([0.5 * (1.0 + math.erf(float(zi) * _INV_SQRT_2)) for zi in np.asarray(z).ravel()]).reshape(np.shape(z))


def _expected_max_of_lines(a: np.ndarray, b: np.ndarray) -> float:
    """
    Closed-form E_Z[ max_j (a_j + b_j Z) ] for Z ~ N(0,1).

    Frazier, Powell & Dayanik (2009), Alg. 1 (upper-envelope of affine
    functions of a scalar standard-normal variable).

    Method: sort lines by slope ascending, drop dominated lines with a
    monotone stack, then sum per-segment integrals in closed form using
        ∫_a^b φ(z) dz = Φ(b) − Φ(a)
        ∫_a^b z φ(z) dz = φ(a) − φ(b)
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = a.size
    if n == 0:
        return 0.0

    # Sort by b ascending; break ties by keeping the line with the largest a.
    order = np.lexsort((-a, b))
    a_s = a[order]
    b_s = b[order]
    # Deduplicate ties in b (keep first occurrence in the sort, which has largest a).
    keep = np.concatenate([[True], np.diff(b_s) > 0])
    a_s = a_s[keep]
    b_s = b_s[keep]
    n = a_s.size

    if n == 1:
        # E[a + b*Z] = a
        return float(a_s[0])

    # Build upper envelope with a monotone stack. C[k] = left boundary of
    # segment k (the intersection of line k with line k-1).
    A: list[float] = [float(a_s[0])]
    B: list[float] = [float(b_s[0])]
    C: list[float] = [-math.inf]

    for i in range(1, n):
        while True:
            # Intersection of top-of-stack line with candidate line i:
            #   A[-1] + B[-1]·z = a[i] + b[i]·z  =>  z = (A[-1] - a[i]) / (b[i] - B[-1])
            # Denominator is > 0 because b is sorted strictly ascending after dedup.
            z_cross = (A[-1] - a_s[i]) / (b_s[i] - B[-1])
            if z_cross <= C[-1]:
                # Top-of-stack line is dominated on its entire active region — pop it.
                A.pop(); B.pop(); C.pop()
                if not A:
                    A.append(float(a_s[i]))
                    B.append(float(b_s[i]))
                    C.append(-math.inf)
                    break
            else:
                A.append(float(a_s[i]))
                B.append(float(b_s[i]))
                C.append(float(z_cross))
                break

    # Add +inf as the right boundary of the last segment.
    C.append(math.inf)
    C_arr = np.array(C)
    A_arr = np.array(A)
    B_arr = np.array(B)

    left = C_arr[:-1]
    right = C_arr[1:]

    # Standard-normal cdf/pdf, using ±inf sentinels
    Phi_left = np.where(np.isinf(left), 0.0, _norm_cdf(np.where(np.isinf(left), 0.0, left)))
    Phi_right = np.where(np.isinf(right), 1.0, _norm_cdf(np.where(np.isinf(right), 0.0, right)))
    phi_left = np.where(np.isinf(left), 0.0, _norm_pdf(np.where(np.isinf(left), 0.0, left)))
    phi_right = np.where(np.isinf(right), 0.0, _norm_pdf(np.where(np.isinf(right), 0.0, right)))

    integral = A_arr * (Phi_right - Phi_left) + B_arr * (phi_left - phi_right)
    return float(integral.sum())


class KGPolicy:
    """
    Proposes the C* that maximally improves the expected posterior-mean minimum.

    Correlated-beliefs KG:

        KG(x) = min_j μ_n(grid_j) − E_z[ min_j (μ_n(grid_j) + w_j(x) · z) ]

    where z ~ N(0,1) and the influence vector is
        w_j(x) = Cov_n(x, grid_j) / √(Var_n(x) + σ²_noise).

    The expectation is computed **analytically** using the upper-envelope
    algorithm of Frazier, Powell & Dayanik (2009), so KG is exact and
    completely deterministic given the observations (no rng dependence).
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        # rng is accepted for AcquisitionPolicy protocol compatibility; unused.
        kg = self._kg_values(model)
        return float(self._grid[np.argmax(kg)])

    def kg_values(self, model: BeliefModel, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return KG for every grid point. rng is accepted but unused."""
        return self._kg_values(model)

    def _kg_values(self, model: BeliefModel) -> np.ndarray:
        grid = self._grid
        m = len(grid)

        mu, _ = model.posterior(grid)                          # (m,)
        current_min = float(np.min(mu))

        post_cov = model.posterior_cov_matrix(grid)            # (m, m)
        post_var = np.maximum(np.diag(post_cov), 0.0)          # (m,)
        noise_var = model.config.noise_std ** 2
        sigma_total = np.sqrt(post_var + noise_var)            # (m,)

        kg = np.zeros(m)
        for i in range(m):
            if sigma_total[i] <= 1e-12:
                # Zero influence — observing here doesn't move any belief.
                continue
            # Influence vector: w_j = Cov(x_i, x_j) / sigma_total[i]
            w = post_cov[i] / sigma_total[i]                   # (m,)
            # E_Z[min_j (mu[j] + w[j] Z)] = -E_Z[max_j (-mu[j] - w[j] Z)]
            e_max = _expected_max_of_lines(-mu, -w)
            expected_future_min = -e_max
            kg[i] = current_min - expected_future_min
        return kg
