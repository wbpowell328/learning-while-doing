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
from typing import Protocol, Sequence, Union

import numpy as np

from .belief import BeliefModel


class AcquisitionPolicy(Protocol):
    """Structural interface satisfied by RandomPolicy, IEPolicy, and KGPolicy."""
    def propose(self, model: BeliefModel, rng: np.random.Generator): ...


@dataclass(frozen=True)
class AcquisitionConfig:
    # For 1D problems: impparam_min/max are floats.
    # For higher-dim problems: pass tuples of length dim.
    # grid_size is PER-DIMENSION; total grid has grid_size**dim points.
    impparam_min: Union[float, Sequence[float]] = 0.01
    impparam_max: Union[float, Sequence[float]] = 0.20
    grid_size: int = 100
    # IE tunable parameter — see IEPolicy docstring.
    z_alpha: float = 0.0
    # RandomizedGreedy tunable parameter — Gaussian noise std added to
    # the argmax-of-belief θ (same units as θ). 0 = pure greedy.
    sigma_greedy: float = 0.0


def _make_grid(lo, hi, grid_size: int) -> np.ndarray:
    """
    Build a flat grid of shape (n_points, dim) over the box [lo, hi].

    * lo, hi may be scalar (dim=1) or length-dim sequences.
    * n_points = grid_size ** dim.
    """
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    if lo.shape != hi.shape:
        raise ValueError(f"impparam_min and impparam_max must have same shape; got {lo.shape} vs {hi.shape}")
    dim = lo.size
    axes = [np.linspace(lo[d], hi[d], grid_size) for d in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)  # (grid_size**dim, dim)


def _theta_dim(cfg: AcquisitionConfig) -> int:
    """Extract the dimension of θ implied by the config's bounds."""
    return int(np.atleast_1d(np.asarray(cfg.impparam_min, dtype=float)).size)


def _grid_pick(grid: np.ndarray, idx: int):
    """
    Return grid[idx] as a scalar float for dim=1 grids, or a length-dim
    numpy array for dim>=2.  Used at the end of every policy's propose().
    """
    row = grid[idx]
    return float(row[0]) if row.size == 1 else row.copy()


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomPolicy:
    """
    Proposes a θ drawn uniformly at random from the box
    [impparam_min, impparam_max].  Works for any dimension.
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        self._lo = np.atleast_1d(np.asarray(self.config.impparam_min, dtype=float))
        self._hi = np.atleast_1d(np.asarray(self.config.impparam_max, dtype=float))
        self._dim = self._lo.size

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        theta = rng.uniform(self._lo, self._hi)
        return float(theta[0]) if self._dim == 1 else theta


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
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        mu, std = model.posterior(self._grid)
        z = self.config.z_alpha
        score = mu - z * std
        return _grid_pick(self._grid, int(np.argmin(score)))


# ---------------------------------------------------------------------------
# Randomized greedy
# ---------------------------------------------------------------------------

class RandomizedGreedyPolicy:
    """
    Pick θ that appears to be best under the current belief, then jitter
    it by N(0, σ_greedy) noise (per-dimension in the multivariate case).
    The belief works in the internal "minimise" frame, so "best" is
    argmin of the posterior mean over the grid.

        θ_next = clip( argmin_θ μ_n(θ) + N(0, σ_greedy²) ,
                       impparam_min, impparam_max )

    σ_greedy = 0 recovers pure greedy exploitation (deterministic argmin
    of the posterior mean). Larger σ_greedy explores the neighborhood of
    the greedy pick without ever fully abandoning it.
    """

    def __init__(self, config: AcquisitionConfig | None = None) -> None:
        self.config = config or AcquisitionConfig()
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)
        self._lo = np.atleast_1d(np.asarray(cfg.impparam_min, dtype=float))
        self._hi = np.atleast_1d(np.asarray(cfg.impparam_max, dtype=float))
        self._dim = self._lo.size

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        mu, _ = model.posterior(self._grid)
        best_row = self._grid[int(np.argmin(mu))]
        sigma = max(0.0, float(self.config.sigma_greedy))
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=self._dim)
            proposal = np.clip(best_row + noise, self._lo, self._hi)
        else:
            proposal = best_row
        return float(proposal[0]) if self._dim == 1 else proposal


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
        w_j(x) = Cov_n(x, grid_j) / √(Var_n(x) + σ²_noise/m*).

    The `m_star` parameter (default 1) scales the effective precision of a
    hypothetical measurement: setting m*>1 evaluates KG as if we were
    going to run m* repeat experiments and average, even though we only
    actually run one. This is the classical "batch-size trick" to
    overcome the S-curve effect — single-shot KG is often near-zero, but
    KG at m* > 1 can be substantial. See kg_vs_m chart for how to pick
    m* per the argmax_m KG(x;m)/m rule.
    """

    def __init__(self, config: AcquisitionConfig | None = None,
                 m_star: int = 1) -> None:
        self.config = config or AcquisitionConfig()
        self._m_star = max(1, int(m_star))
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        """Change the effective-batch-size scaling used for KG evaluation."""
        self._m_star = max(1, int(m_star))

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        # rng accepted for protocol compatibility; unused.
        kg = self._kg_values(model)
        return _grid_pick(self._grid, int(np.argmax(kg)))

    def kg_values(self, model: BeliefModel, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return KG for every grid point. rng is accepted but unused."""
        return self._kg_values(model)

    def _kg_values(self, model: BeliefModel) -> np.ndarray:
        grid = self._grid                           # (m, dim)
        m = grid.shape[0]

        mu, _ = model.posterior(grid)                          # (m,)
        current_min = float(np.min(mu))

        post_cov = model.posterior_cov_matrix(grid)            # (m, m)
        post_var = np.maximum(np.diag(post_cov), 0.0)          # (m,)
        # Noise scaled by 1/m*: treat the observation as m* averaged reps.
        noise_var = (model.config.noise_std ** 2) / float(self._m_star)
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

def _kg_pre_compute(model: BeliefModel, grid: np.ndarray, candidates: np.ndarray,
                    m_star: int = 1):
    """
    Common pre-computation for the KG-at-candidates helpers.

    `m_star` (default 1) scales the effective precision of a hypothetical
    measurement — the noise variance term is divided by m_star. m_star>1
    corresponds to the classical "batch-size trick" (evaluate KG as if we
    were going to average m_star repeat observations at the candidate).

    Returns
    -------
    mu_grid       : (m,)  posterior mean at grid points
    cand_mean     : (n_c,) posterior mean at candidates
    cand_var      : (n_c,) posterior variance at candidates
    cross_cov     : (n_c, m) posterior covariance Cov_n(candidate_i, grid_j)
    sigma_tilde   : (n_c,) sqrt(cand_var + noise_var/m_star)
    delta_at_cand : (n_c,) cand_var / sigma_tilde
                      (= Cov_n(x_i, x_i) / sigma_tilde_i — Bayesian mean-shift std at x_i itself)
    """
    grid = np.asarray(grid, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    # Ensure both are 2-D (n, dim); auto-reshape 1-D to (n, 1).
    if grid.ndim == 1:
        grid = grid.reshape(-1, 1)
    if candidates.ndim == 1:
        candidates = candidates.reshape(-1, 1)
    m = grid.shape[0]
    n_c = candidates.shape[0]

    mu_grid, _ = model.posterior(grid)
    cand_mean, _ = model.posterior(candidates)

    # Full posterior covariance on the union {candidates, grid} — slice as needed.
    all_pts = np.concatenate([candidates, grid], axis=0)
    full_cov = model.posterior_cov_matrix(all_pts)
    cand_cov = full_cov[:n_c, :n_c]
    cross_cov = full_cov[:n_c, n_c:]

    cand_var = np.maximum(np.diag(cand_cov), 0.0)
    noise_var = (model.config.noise_std ** 2) / max(1, int(m_star))
    sigma_tilde = np.sqrt(cand_var + noise_var)
    safe_sigma = np.where(sigma_tilde > 1e-12, sigma_tilde, 1.0)
    delta_at_cand = cand_var / safe_sigma  # zero where sigma_tilde ≈ 0

    return mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta_at_cand


def _std_normal_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


# math.erf is scalar; numpy has no erf without scipy. Vectorise once.
_np_erf = np.vectorize(__import__('math').erf, otypes=[float])


def _std_normal_cdf(z):
    return 0.5 * (1.0 + _np_erf(z / np.sqrt(2.0)))


def kg_indep_beliefs_vs_batch_size(model: BeliefModel, grid: np.ndarray,
                                    theta, m_values,
                                    history: list,
                                    future_noise_std: float | None = None) -> np.ndarray:
    """
    Classical INDEPENDENT-BELIEFS KG(m) at a candidate θ. Correlations
    across θ are dropped entirely: each grid point is a separate
    alternative with its own conjugate Normal-Normal belief, updated
    only by observations that fell in its own bin (nearest-grid-point
    assignment). Then apply the classical single-alternative KG formula.

    Uses model.config for signal_std (prior amplitude), noise_std (σ_ε)
    and prior_mean. `history` is the session's observation list in the
    belief's *internal* (cost) frame — the caller (shell/app.py) is
    responsible for the sign flip.

    This is the setting from Frazier & Powell (OSPL, independent-normal-
    beliefs chapter) that produces the pedagogically-classical S-curve
    when |Δ| is comparable to σ̃(m). Contrast with the CORRELATED KG
    that the policy actually uses (smoother upper envelope) or with the
    hybrid kg_indep_scalar_vs_batch_size (independent-KG formula, but
    on the GP-smoothed posterior).
    """
    cfg = model.config
    prior_var  = float(cfg.signal_std) ** 2
    noise_var  = float(cfg.noise_std) ** 2
    prior_mean = float(cfg.prior_mean)
    if prior_var <= 0 or noise_var <= 0:
        return np.zeros(len(list(m_values)))

    # Grid to (n_grid, dim). Works for both 1-D (n_grid, 1) and 2-D+ cases.
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.ndim == 1:
        grid_arr = grid_arr.reshape(-1, 1)
    n_grid, dim = grid_arr.shape

    # Bin every observation into its nearest grid cell (Euclidean distance).
    n_obs   = np.zeros(n_grid, dtype=float)
    sum_obs = np.zeros(n_grid, dtype=float)
    for theta_hist, val_hist in history:
        t_vec = np.atleast_1d(np.asarray(theta_hist, dtype=float)).reshape(-1)
        if t_vec.size != dim:
            continue    # skip malformed history rows rather than crash
        dists = np.linalg.norm(grid_arr - t_vec, axis=1)
        j = int(np.argmin(dists))
        n_obs[j]   += 1.0
        sum_obs[j] += float(val_hist)

    # Independent conjugate Normal-Normal update at each grid cell:
    #   precision_j = 1/prior_var + n_j/noise_var
    #   V_j         = 1/precision_j
    #   μ_j         = V_j · (prior_mean/prior_var + Σ y_i / noise_var)
    prec_j = 1.0 / prior_var + n_obs / noise_var
    V_j    = 1.0 / prec_j
    mu_j   = V_j * (prior_mean / prior_var + sum_obs / noise_var)

    # Snap the candidate θ to the nearest grid cell.
    theta_vec = np.atleast_1d(np.asarray(theta, dtype=float)).reshape(-1)
    if theta_vec.size != dim:
        return np.zeros(len(list(m_values)))
    dists_cand = np.linalg.norm(grid_arr - theta_vec, axis=1)
    j_star = int(np.argmin(dists_cand))
    V_x  = float(V_j[j_star])
    mu_x = float(mu_j[j_star])

    # Current best = min μ over the OTHER grid alternatives.
    mask = np.ones(n_grid, dtype=bool); mask[j_star] = False
    if not mask.any():
        return np.zeros(len(list(m_values)))
    mu_best = float(np.min(mu_j[mask]))
    delta_abs = abs(mu_x - mu_best)

    # `future_noise_std` semantics: this is the noise Warren asks us to
    # assume for HYPOTHETICAL batch experiments at the candidate. The
    # historical posterior above stays fit under model.config.noise_std
    # (the belief's actual noise assumption) — this is what makes the
    # classical S visible: Δ stays substantial (from real data) while
    # σ̃(m=1) collapses as future_noise_std grows.
    future_noise_var = (float(future_noise_std) ** 2
                        if future_noise_std is not None else noise_var)
    m_arr = np.asarray(list(m_values), dtype=float)
    denom = V_x + future_noise_var / np.maximum(m_arr, 1e-12)
    var_shift = (V_x ** 2) / np.maximum(denom, 1e-30)
    st = np.sqrt(np.maximum(var_shift, 0.0))
    safe_st = np.where(st > 1e-12, st, 1.0)
    z = -delta_abs / safe_st
    # Classical single-alternative KG formula (Frazier & Powell OSPL):
    #     f(z) = φ(z) + z·Φ(z)
    # At z = 0 (candidate ties current best): f = φ(0) ≈ 0.399.
    # At z → −∞ (candidate hopelessly worse): f → 0 — this is the vanishing
    # regime that produces the classical S when Δ/σ̃ is large.
    f = _std_normal_pdf(z) + z * _std_normal_cdf(z)
    return st * f


def kg_indep_scalar_vs_batch_size(model: BeliefModel, grid: np.ndarray,
                                   theta, m_values) -> np.ndarray:
    """
    Independent-scalar KG at a single candidate θ as a function of the
    batch size m of repeat observations there.

    Uses the classical single-alternative KG formula (Frazier & Powell
    OSPL): treats the candidate as one alternative competing against the
    current best on the grid, ignores cross-θ covariance. This is the KG
    that exhibits the pedagogically-classical S-curve as a function of
    precision (equivalently, of m):

        σ̃(m) = √( V² / (V + σ²/m) ),   V = posterior variance at θ
        Δ     = μ(θ) − μ(best)          (best = argmin_{j≠candidate} μ_j)
        KG(m) = σ̃(m) · f( −|Δ| / σ̃(m) ),
        f(z) = φ(z) + z · Φ(z).

    Correlated KG (the version the KG policy uses) is smoother in m: the
    upper-envelope over many nearby grid points averages out the tight
    threshold that produces the classical S. Both are valid views — this
    one is the pedagogical one Warren asked for. Correlated stays
    available via kg_vs_batch_size for callers that want it.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float)).reshape(1, -1)
    mu_x_arr, std_x_arr = model.posterior(theta_arr)
    mu_x = float(mu_x_arr[0])
    var_x = float(std_x_arr[0]) ** 2

    mu_grid, _ = model.posterior(grid)
    mu_best = float(np.min(mu_grid))    # belief-frame (always minimising)
    delta_abs = abs(mu_x - mu_best)

    sigma_eps = float(model.config.noise_std)
    m_arr = np.asarray(list(m_values), dtype=float)
    out = np.zeros_like(m_arr, dtype=float)
    if var_x <= 0 or sigma_eps <= 0:
        return out

    # Bayesian mean-shift std for m repeat obs at candidate.
    denom = var_x + (sigma_eps ** 2) / np.maximum(m_arr, 1e-12)
    var_shift = (var_x ** 2) / denom
    st = np.sqrt(np.maximum(var_shift, 0.0))
    safe_st = np.where(st > 1e-12, st, 1.0)
    z = -delta_abs / safe_st
    # f(z) = phi(z) - z * Phi(z); nonneg for all z; large for z near 0.
    # Classical single-alternative KG formula (Frazier & Powell OSPL):
    #     f(z) = φ(z) + z·Φ(z)
    # At z = 0 (candidate ties current best): f = φ(0) ≈ 0.399.
    # At z → −∞ (candidate hopelessly worse): f → 0 — this is the vanishing
    # regime that produces the classical S when Δ/σ̃ is large.
    f = _std_normal_pdf(z) + z * _std_normal_cdf(z)
    return st * f


def kg_vs_batch_size(model: BeliefModel, grid: np.ndarray,
                     theta, m_values,
                     noise_std_override: float | None = None) -> np.ndarray:
    """
    KG at a single θ as a function of the effective precision m·β, where β =
    1/σ²_noise and m = number of repeat observations at the same θ. m = 1
    recovers the standard single-experiment analytic KG at that θ; larger m
    behaves as if we ran m repeats and averaged them (noise variance
    reduced by factor m, i.e. σ_noise → σ_noise/√m).

    Motivation: the S-curve effect. A single experiment can produce a tiny
    value of information when we're on the flat lower portion of the S; a
    batch of m experiments moves us up onto the steep part. Plotting
    KG(θ; m) exposes when a policy is "trapped" by single-shot KG being
    small even though the true information at that θ is large.

    Parameters
    ----------
    model : BeliefModel
    grid  : (m_grid, dim) search grid — the same grid the KG policy uses.
    theta : the single candidate θ (scalar for 1-D, length-dim vector otherwise).
    m_values : iterable of positive ints.

    Returns
    -------
    (len(m_values),) ndarray of KG values in belief-frame units.
    """
    # _kg_pre_compute expects a batch of candidates; wrap the single θ.
    cand_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    if cand_arr.ndim == 1 and grid.ndim > 1 and grid.shape[1] > 1:
        cand_arr = cand_arr.reshape(1, -1)                     # multi-d single point
    elif cand_arr.ndim == 1:
        cand_arr = cand_arr.reshape(-1, 1)                     # 1-D → column
    mu_grid, cand_mean, cand_var, cross_cov, _, _ = _kg_pre_compute(
        model, grid, cand_arr
    )
    # Single candidate expected.
    v = float(cand_var[0])
    mu_ext_base = np.concatenate([mu_grid, [cand_mean[0]]])
    cross = cross_cov[0]                                        # (m_grid,)
    current_min = float(np.min(mu_ext_base))

    # noise_std_override lets pedagogy tools ask "what if σ_ε were X?"
    # without refitting the posterior — the posterior mean and covariance
    # (computed from the actual belief) stay fixed, only the future
    # observation-noise term in the KG formula changes.
    sigma_eps = (float(noise_std_override) if noise_std_override is not None
                 else float(model.config.noise_std))
    base_noise_var = sigma_eps ** 2

    m_arr = np.asarray(list(m_values), dtype=int)
    out = np.zeros(m_arr.shape[0], dtype=float)
    for k, m in enumerate(m_arr):
        if m < 1:
            continue
        noise_var_m = base_noise_var / float(m)
        sigma_tilde_m = float(np.sqrt(v + noise_var_m))
        if sigma_tilde_m <= 1e-12:
            continue
        w_ext = np.concatenate([cross / sigma_tilde_m, [v / sigma_tilde_m]])
        e_max = _expected_max_of_lines(-mu_ext_base, -w_ext)
        expected_future_min = -float(e_max)
        out[k] = current_min - expected_future_min
    return out


def kg_analytic_correlated_at(model: BeliefModel, grid: np.ndarray,
                               candidates: np.ndarray,
                               m_star: int = 1) -> np.ndarray:
    """
    Analytic correlated-beliefs KG at each candidate x_i.

    Search set is {grid ∪ x_i}: the candidate itself is one more alternative
    that could become the argmin after observation. `m_star` (default 1)
    scales the effective precision of the hypothetical observation.
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates, m_star=m_star,
    )
    n_c = cand_mean.shape[0]
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
                         rng: np.random.Generator,
                         m_star: int = 1) -> np.ndarray:
    """
    Monte-Carlo correlated-beliefs KG at each candidate — for diagnostic
    comparison against the analytic value.

    n_mc draws of Z ~ N(0,1) are shared across candidates for maximum
    variance reduction (each candidate sees the same Z). `m_star` scales
    the effective precision (see _kg_pre_compute).
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates, m_star=m_star,
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
    the analytic gold standard. Uses `rng` for reproducible MC draws.
    `m_star` (default 1) — see KGPolicy for semantics.
    """
    def __init__(self, config: AcquisitionConfig | None = None, n_mc: int = 500,
                 m_star: int = 1) -> None:
        self.config = config or AcquisitionConfig()
        self._n_mc = int(n_mc)
        self._m_star = max(1, int(m_star))
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        self._m_star = max(1, int(m_star))

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        kg = kg_mc_correlated_at(model, self._grid, self._grid, self._n_mc, rng,
                                 m_star=self._m_star)
        return _grid_pick(self._grid, int(np.argmax(kg)))


class KGIndependentPolicy:
    """
    Offline KG under the independent-beliefs assumption: observing at x
    only shifts belief at x itself (no cascade to correlated neighbors).
    argmax of the resulting closed-form KG picks the next measurement.
    `m_star` (default 1) — see KGPolicy for semantics.
    """
    def __init__(self, config: AcquisitionConfig | None = None,
                 m_star: int = 1) -> None:
        self.config = config or AcquisitionConfig()
        self._m_star = max(1, int(m_star))
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        self._m_star = max(1, int(m_star))

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        kg = kg_independent_at(model, self._grid, self._grid, m_star=self._m_star)
        return _grid_pick(self._grid, int(np.argmax(kg)))


class OKGCorrelatedPolicy:
    """
    Online KG with correlated beliefs — Warren-2026 formulation.

        OKG(x) = mu_n(x) - offline_correlated_KG(x; m*)          [cost frame]
        argmax_x [mu^n(x) + KG(x; m*)]                            [reward frame]

    Replaces the classical Ryzhov (N-n)·KG multiplier with a
    user-tunable m* (units: days = expected batch size of a
    hypothetical replicate). m* scales the effective precision of a
    hypothetical measurement inside KG, so the info-value bonus grows
    with m* without depending on (N,n).
    """
    def __init__(self, config: AcquisitionConfig,
                 m_star: int = 1, **_unused) -> None:
        # `**_unused` swallows legacy kwargs (e.g. budget=...) from any
        # caller that hasn't been migrated yet — safe no-op.
        self.config = config
        self._m_star = max(1, int(m_star))
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        self._m_star = max(1, int(m_star))

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        mu, _ = model.posterior(self._grid)
        kg = kg_analytic_correlated_at(model, self._grid, self._grid, m_star=self._m_star)
        okg = mu - kg
        return _grid_pick(self._grid, int(np.argmin(okg)))


class OKGRyzhovCorrelatedPolicy:
    """
    Classical Ryzhov online KG (correlated beliefs) with the (N − n)
    multiplier on the single-shot offline KG term:

        OKG_ryzhov(x) = μ_n(x) − (N − n) · offline_correlated_KG(x; m=1)  [cost frame]
        argmax_x [ μ_reward(x) + (N − n) · offline_correlated_KG(x; m=1) ] [reward frame]

    The lookahead multiplier is fixed at m=1 (single-shot KG) — Ryzhov's
    only tuning knob is N. (The separate lookahead knob Hᵒⁿ lives on the
    online-correlated policy OKGCorrelatedPolicy, not here.) set_m_star is
    kept as an accepted no-op so generic KG-policy plumbing that pushes a
    lookahead value doesn't have to special-case Ryzhov.

    N is the session's `budget`; n comes from BeliefModel.n_observations
    at propose time.
    """
    def __init__(self, config: AcquisitionConfig,
                 budget: int | None = 10, m_star: int = 1, **_unused) -> None:
        self.config = config
        # Default N=10 if the caller didn't supply one — matches the
        # legacy /kg endpoint's default_budget for display continuity.
        self._N = max(1, int(budget)) if budget is not None else 10
        # Fixed at 1 by design; kept as an attribute only so introspection
        # (e.g. the KG chart reading pol._m_star) sees a consistent value.
        self._m_star = 1
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        # No-op: Ryzhov always uses single-shot KG (m=1). Accepting the call
        # keeps set_policy()/Session.__init__ plumbing uniform across policies.
        self._m_star = 1

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        mu, _ = model.posterior(self._grid)
        kg = kg_analytic_correlated_at(model, self._grid, self._grid, m_star=1)
        n = int(model.n_observations)
        remaining = max(0, self._N - n)
        okg = mu - remaining * kg
        return _grid_pick(self._grid, int(np.argmin(okg)))


class OKGIndependentPolicy:
    """
    Online KG using independent-beliefs offline KG as the info-value
    term. Same functional form as OKGCorrelatedPolicy — the (N-n)
    multiplier is gone; m* controls exploration.
    """
    def __init__(self, config: AcquisitionConfig,
                 m_star: int = 1, **_unused) -> None:
        self.config = config
        self._m_star = max(1, int(m_star))
        cfg = self.config
        self._grid = _make_grid(cfg.impparam_min, cfg.impparam_max, cfg.grid_size)

    def set_m_star(self, m_star: int) -> None:
        self._m_star = max(1, int(m_star))

    def propose(self, model: BeliefModel, rng: np.random.Generator):
        mu, _ = model.posterior(self._grid)
        kg = kg_independent_at(model, self._grid, self._grid, m_star=self._m_star)
        okg = mu - kg
        return _grid_pick(self._grid, int(np.argmin(okg)))


def kg_independent_at(model: BeliefModel, grid: np.ndarray,
                       candidates: np.ndarray,
                       m_star: int = 1) -> np.ndarray:
    """
    Independent-beliefs KG at each candidate — closed form.

    Pretends observing at candidate x_i shifts the belief only at x_i itself
    (no correlation cascade to grid points).  With M = min_j μ_n(grid_j) fixed
    and future μ(x_i) = μ_n(x_i) + δ_i·Z, δ_i = Var_n(x_i)/σ̃_i, and letting
    a_i = M − μ_n(x_i), t_i = a_i/δ_i:

        KG_indep(x_i) = min(M, μ_n(x_i)) − (M − a_i·Φ(t_i) − δ_i·φ(t_i))

    `m_star` (default 1) scales the effective precision — the noise
    variance term in σ̃ is divided by m_star.
    """
    mu_grid, cand_mean, cand_var, cross_cov, sigma_tilde, delta = _kg_pre_compute(
        model, grid, candidates, m_star=m_star,
    )
    M = float(np.min(mu_grid))
    n_c = cand_mean.shape[0]
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
