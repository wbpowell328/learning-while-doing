"""
Acquisition policies for the learning-while-doing optimizer.

Each policy implements one method:
    propose(model: BeliefModel, rng: np.random.Generator) -> float

returning the next θ to evaluate.  Policies are stateless given
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
    # IE (interval-estimation / lower confidence bound) tunable parameter.
    #   IE(x) = mu(x) - z_alpha * std(x); pick argmin.
    # z_alpha = 0    → pure exploitation (argmin posterior mean)
    # z_alpha → inf  → pure exploration (argmax posterior std)
    z_alpha: float = 0.0


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomPolicy:
    """
    Proposes a θ drawn uniformly at random from [c_star_min, c_star_max].
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
    Interval-estimation / lower-confidence-bound acquisition (for minimization):

        IE(x) = mu(x) - z_alpha * std(x)

    Picks argmin.  z_alpha (in AcquisitionConfig) is the exploration knob:

        z_alpha = 0     — pure exploitation: argmin posterior mean.
        z_alpha ~ 1-2   — mild exploration bonus for uncertain points.
        z_alpha → inf   — pure exploration: argmax posterior std.

    With no observations the prior mean and std are both uniform, so the first
    proposal is grid[0] regardless of z_alpha (deterministic tiebreak).
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        mu, std = model.posterior(self._grid)
        z = self.config.z_alpha
        score = mu - z * std
        return float(self._grid[np.argmin(score)])


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
    Proposes the θ that maximally improves the expected posterior-mean minimum.

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


# ---------------------------------------------------------------------------
# KG at arbitrary candidate points — for pedagogical comparison plots.
#
# These helpers compute KG at any set of candidate θ values, using a fixed
# search grid.  The search set for the KG calculation is the union of the
# fixed grid and the candidate — so the candidate itself is treated as one
# more alternative that could become the "best" after observation.
#
# Three flavors:
#   analytic-correlated (exact, via FPD 2009 upper-envelope)
#   monte-carlo-correlated (for direct MC-vs-analytic diagnostic)
#   independent           (pretends observing at x only shifts belief at x)
# ---------------------------------------------------------------------------

def _kg_pre_compute(model: BeliefModel, grid: np.ndarray, candidates: np.ndarray):
    """
    Common pre-computation for the three KG-at-candidates helpers.

    Returns
    -------
    mu_grid       : (m,)  posterior mean at grid points
    cand_mean     : (n_c,) posterior mean at candidates
    cand_var      : (n_c,) posterior variance at candidates
    cross_cov     : (n_c, m) posterior covariance Cov_n(candidate_i, grid_j)
    sigma_tilde   : (n_c,) sqrt(cand_var + noise_var)
    delta_at_cand : (n_c,) cand_var / sigma_tilde
                      (= Cov_n(x_i, x_i) / sigma_tilde_i — Bayesian mean-shift std at x_i itself)
    """
    grid = np.asarray(grid, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    m = grid.size
    n_c = candidates.size

    mu_grid, _ = model.posterior(grid)
    cand_mean, _ = model.posterior(candidates)

    # Full posterior covariance on the union {candidates, grid} — slice as needed.
    all_pts = np.concatenate([candidates, grid])
    full_cov = model.posterior_cov_matrix(all_pts)
    cand_cov = full_cov[:n_c, :n_c]
    cross_cov = full_cov[:n_c, n_c:]

    cand_var = np.maximum(np.diag(cand_cov), 0.0)
    noise_var = model.config.noise_std ** 2
    sigma_tilde = np.sqrt(cand_var + noise_var)
    safe_sigma = np.where(sigma_tilde > 1e-12, sigma_tilde, 1.0)
    delta_at_cand = cand_var / safe_sigma  # zero where sigma_tilde ≈ 0

    return mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta_at_cand


def kg_analytic_correlated_at(model: BeliefModel, grid: np.ndarray,
                               candidates: np.ndarray) -> np.ndarray:
    """
    Analytic correlated-beliefs KG at each candidate x_i.

    Search set is {grid ∪ x_i}: the candidate itself is one more alternative
    that could become the argmin after observation.
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates
    )
    n_c = candidates.size
    kg = np.zeros(n_c)
    for i in range(n_c):
        if sigma_tilde[i] <= 1e-12:
            continue
        mu_ext = np.concatenate([mu_grid, [cand_mean[i]]])
        w_ext = np.concatenate([cross_cov[i] / sigma_tilde[i], [delta[i]]])
        current_min = float(np.min(mu_ext))
        e_max = _expected_max_of_lines(-mu_ext, -w_ext)
        expected_future_min = -e_max
        kg[i] = current_min - expected_future_min
    return kg


def kg_mc_correlated_at(model: BeliefModel, grid: np.ndarray,
                         candidates: np.ndarray, n_mc: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Monte-Carlo correlated-beliefs KG at each candidate — for diagnostic
    comparison against the analytic value.

    n_mc draws of Z ~ N(0,1) are shared across candidates for maximum
    variance reduction (each candidate sees the same Z).
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates
    )
    n_c = candidates.size
    z = rng.standard_normal(n_mc)  # (n_mc,)
    kg = np.zeros(n_c)
    for i in range(n_c):
        if sigma_tilde[i] <= 1e-12:
            continue
        mu_ext = np.concatenate([mu_grid, [cand_mean[i]]])
        w_ext = np.concatenate([cross_cov[i] / sigma_tilde[i], [delta[i]]])
        current_min = float(np.min(mu_ext))
        # future_min[k] = min_j (mu_ext[j] + w_ext[j] * z[k])
        future_min = np.min(mu_ext[None, :] + w_ext[None, :] * z[:, None], axis=1)
        kg[i] = current_min - float(np.mean(future_min))
    return kg


class KGMCPolicy:
    """
    Same estimand as KGPolicy but computed with Monte-Carlo. Useful as a
    baseline to see how MC noise degrades KG-driven decisions relative to
    the analytic gold standard.

    Uses `rng` for reproducible MC draws.
    """
    def __init__(self, config: AcquisitionConfig | None = None, n_mc: int = 500) -> None:
        self.config = config or AcquisitionConfig()
        self._n_mc = int(n_mc)
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        kg = kg_mc_correlated_at(model, self._grid, self._grid, self._n_mc, rng)
        return float(self._grid[np.argmax(kg)])


class KGIndependentPolicy:
    """
    Offline KG under the independent-beliefs assumption: observing at x
    only shifts belief at x itself (no cascade to correlated neighbors).
    argmax of the resulting closed-form KG picks the next measurement.
    """
    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        kg = kg_independent_at(model, self._grid, self._grid)
        return float(self._grid[np.argmax(kg)])


class OKGCorrelatedPolicy:
    """
    Online KG (Ryzhov 2010) with correlated beliefs.

        OKG(x) = mu_n(x) - (N - n) * offline_correlated_KG(x)

    where N is the total measurement budget and n is the number of steps
    taken so far (read from model.n_observations). Picks argmin OKG.
    """
    def __init__(self, config: AcquisitionConfig, budget: int) -> None:
        self.config = config
        self._budget = int(budget)
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        n = model.n_observations
        remaining = max(0, self._budget - n)
        mu, _ = model.posterior(self._grid)
        kg = kg_analytic_correlated_at(model, self._grid, self._grid)
        okg = mu - remaining * kg
        return float(self._grid[np.argmin(okg)])


class OKGIndependentPolicy:
    """
    Online KG using independent-beliefs offline KG as the info-value term.
    Same functional form as OKGCorrelatedPolicy but with the independent
    KG plugged in.
    """
    def __init__(self, config: AcquisitionConfig, budget: int) -> None:
        self.config = config
        self._budget = int(budget)
        cfg = self.config
        self._grid = np.linspace(cfg.c_star_min, cfg.c_star_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator) -> float:
        n = model.n_observations
        remaining = max(0, self._budget - n)
        mu, _ = model.posterior(self._grid)
        kg = kg_independent_at(model, self._grid, self._grid)
        okg = mu - remaining * kg
        return float(self._grid[np.argmin(okg)])


def kg_independent_at(model: BeliefModel, grid: np.ndarray,
                       candidates: np.ndarray) -> np.ndarray:
    """
    Independent-beliefs KG at each candidate — closed form.

    Pretends observing at candidate x_i shifts the belief only at x_i itself
    (no correlation cascade to grid points).  With M = min_j μ_n(grid_j) fixed
    and future μ(x_i) = μ_n(x_i) + δ_i·Z, δ_i = Var_n(x_i)/σ̃_i, and letting
    a_i = M − μ_n(x_i), t_i = a_i/δ_i:

        KG_indep(x_i) = min(M, μ_n(x_i)) − (M − a_i·Φ(t_i) − δ_i·φ(t_i))
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates
    )
    M = float(np.min(mu_grid))
    n_c = candidates.size
    kg = np.zeros(n_c)
    for i in range(n_c):
        if delta[i] <= 1e-12:
            continue
        a = M - float(cand_mean[i])
        t = a / delta[i]
        # _norm_cdf and _norm_pdf both work on numpy arrays; wrap scalar in array.
        Phi_t = float(_norm_cdf(np.array([t]))[0])
        phi_t = float(_norm_pdf(np.array([t]))[0])
        current_min = min(M, float(cand_mean[i]))
        expected_future_min = M - a * Phi_t - delta[i] * phi_t
        kg[i] = current_min - expected_future_min
    return kg
