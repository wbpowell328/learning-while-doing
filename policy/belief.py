"""
Gaussian Process belief model over F(θ) = E[cost | θ].

Kernel: squared-exponential (RBF) with per-dimension length scales (ARD):
    k(x, x') = signal_std² × exp(-0.5 × Σ_d ((x_d - x'_d)² / ℓ_d²))

Observations y_i = F(θ_i) + ε_i, ε_i ~ N(0, noise_std²).

The posterior is recomputed from scratch on each query after a new observation
(lazy Cholesky cache, invalidated on update).  Correct and simple; incremental
Cholesky updates can be added if n_observations grows large.

Dimensionality
--------------
`BeliefModel(config, dim=d)` supports any input dimension d ≥ 1.  Internally
all inputs are 2-D arrays of shape (n, d).  For d=1, callers may pass flat
arrays of shape (n,) — the model auto-reshapes.  `length_scale` on the config
may be a scalar (broadcast to all d dimensions) or a length-d array (ARD).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BeliefConfig:
    # RBF kernel hyperparameters.
    # length_scale: scalar → same across all dims (isotropic);
    #               sequence of length d → per-dimension (ARD).
    length_scale: float | Sequence[float] = 0.04
    # Per-day units. The Session divides observations by n_days before
    # calling belief.update, so signal_std / noise_std / prior_mean all
    # describe the daily reward frame rather than a batch total. This
    # keeps the belief coherent across mixed batch lengths.
    signal_std: float = 25.0       # prior amplitude of F(θ), $/day
    noise_std:  float = 100.0      # per-day observation noise, $/day
    prior_mean: float = 175.0      # expected daily reward before any data
    jitter:     float = 1e-6      # diagonal jitter for Cholesky stability


def _as_2d(x, dim: int) -> np.ndarray:
    """Coerce a scalar/1-D/2-D input into shape (n, dim)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:                      # scalar
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if dim == 1:
            arr = arr.reshape(-1, 1)       # flat (n,) → (n, 1)
        elif arr.size == dim:
            arr = arr.reshape(1, dim)      # single vector (d,) → (1, d)
        else:
            raise ValueError(f"cannot reshape 1-D array of size {arr.size} to (?, {dim})")
    elif arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(f"expected shape (n, {dim}); got {arr.shape}")
    else:
        raise ValueError(f"expected 0/1/2-D input; got {arr.ndim}-D")
    return arr


class BeliefModel:
    """
    GP surrogate for E[cost | θ] with support for scalar or vector θ.

    Usage
    -----
        model = BeliefModel(BeliefConfig(), dim=2)
        model.update([0.05, 0.20], 3000.0)
        mean, std = model.posterior(query_grid)   # query_grid: (m, 2) or (2,)
    """

    def __init__(self, config: BeliefConfig | None = None, dim: int = 1) -> None:
        self.config: BeliefConfig = config if config is not None else BeliefConfig()
        self.dim: int = int(dim)

        # Per-dim length scales.  Scalar → broadcast; sequence → ARD.
        ls = np.asarray(self.config.length_scale, dtype=float).reshape(-1)
        if ls.size == 1:
            self._length_scales = np.full(self.dim, float(ls[0]))
        elif ls.size == self.dim:
            self._length_scales = ls.copy()
        else:
            raise ValueError(
                f"length_scale has {ls.size} value(s) but dim={self.dim}"
            )

        # Observation store — always 2-D internally: X shape (n, dim), y shape (n,).
        self._X: np.ndarray = np.empty((0, self.dim), dtype=float)
        self._y: np.ndarray = np.empty((0,), dtype=float)
        # Per-observation noise variance. Empty means "use config.noise_std²
        # for every row" (backward-compat with callers that don't pass one).
        # Populated by Session when observations come from different-length
        # batches so the belief can trust longer batches more (variance of
        # a per-day-average scales as σ² / n_days).
        self._obs_noise_var: np.ndarray = np.empty((0,), dtype=float)
        self._chol:  np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, impparam, observed_cost: float,
               noise_std: float | None = None) -> None:
        """
        Append one (θ, cost) observation.

        `noise_std` (optional): if given, overrides config.noise_std for
        just this observation. Session uses this so a per-day-average over
        n days is told to the belief with noise σ_n / √n_days (avg-of-iid
        scaling), instead of the belief conservatively assuming single-
        day noise on every observation.
        """
        x = _as_2d(impparam, self.dim)
        if x.shape[0] != 1:
            raise ValueError(f"update() expects a single point; got shape {x.shape}")
        self._X = np.vstack([self._X, x])
        self._y = np.concatenate([self._y, [float(observed_cost)]])
        var = float(noise_std) ** 2 if noise_std is not None else float(self.config.noise_std) ** 2
        self._obs_noise_var = np.concatenate([self._obs_noise_var, [var]])
        self._chol = None
        self._alpha = None

    def posterior(self, impparams) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) at each query point.

        impparams: array-like of shape (m, dim), or (m,) if dim==1, or (dim,)
        for a single point.  Returns two length-m arrays.
        """
        X_q = _as_2d(impparams, self.dim)
        cfg = self.config

        if self._X.shape[0] == 0:
            mean = np.full(X_q.shape[0], cfg.prior_mean)
            std  = np.full(X_q.shape[0], cfg.signal_std)
            return mean, std

        self._build_cache()

        K_star_X = self._rbf(X_q, self._X)                # (m, n)
        mean = cfg.prior_mean + K_star_X @ self._alpha    # (m,)

        # Var(x) = k(x, x) - k(X, x)^T (K + σ² I)^{-1} k(X, x)
        # k(x, x) = signal_std² for the RBF kernel.
        V = np.linalg.solve(self._chol, K_star_X.T)       # (n, m)
        post_var = cfg.signal_std ** 2 - np.sum(V ** 2, axis=0)
        std = np.sqrt(np.maximum(post_var, 0.0))
        return mean, std

    def posterior_at(self, impparam) -> tuple[float, float]:
        """Convenience wrapper: return (mean, std) at a single point."""
        m, s = self.posterior(impparam)
        return float(m[0]), float(s[0])

    def posterior_cov_matrix(self, impparams) -> np.ndarray:
        """
        Full posterior covariance matrix at the query points, shape (m, m).

        Cov_n(x_i, x_j) = k(x_i, x_j) − V_i · V_j
        where V = L⁻¹ k(X, x*), L = chol(K(X,X) + σ² I).
        With no observations this equals the prior covariance k(x*, x*).
        """
        X_q = _as_2d(impparams, self.dim)
        K_star = self._rbf(X_q, X_q)                      # (m, m)

        if self._X.shape[0] == 0:
            return K_star

        self._build_cache()
        k_X_grid = self._rbf(self._X, X_q)                # (n, m)
        V = np.linalg.solve(self._chol, k_X_grid)         # (n, m)
        return K_star - V.T @ V                           # (m, m)

    @property
    def n_observations(self) -> int:
        return int(self._X.shape[0])

    @property
    def observations(self) -> tuple[list, list]:
        """
        Return (impparams, costs) as lists — dim=1 returns flat floats,
        higher dims return list-of-lists.
        """
        if self.dim == 1:
            return list(self._X[:, 0]), list(self._y)
        return [row.tolist() for row in self._X], list(self._y)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rbf(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        ARD squared-exponential kernel.

        X1: (m, dim), X2: (n, dim).  Returns (m, n).
        """
        cfg = self.config
        # Broadcast to (m, n, dim); normalize each dim by its own length scale.
        diffs = X1[:, None, :] - X2[None, :, :]
        scaled = diffs / self._length_scales
        sq_dist = np.sum(scaled ** 2, axis=-1)
        return cfg.signal_std ** 2 * np.exp(-0.5 * sq_dist)

    def _build_cache(self) -> None:
        """Cache Cholesky of K(X,X) + σ² I and the weight vector α."""
        if self._chol is not None:
            return
        cfg = self.config
        n = self._X.shape[0]

        K = self._rbf(self._X, self._X)
        # Per-observation noise variance on the diagonal (falls back to
        # config.noise_std² when a row was appended without an override).
        noise_var = self._obs_noise_var if self._obs_noise_var.size == n \
                    else np.full(n, cfg.noise_std ** 2)
        K += np.diag(noise_var + cfg.jitter)

        self._chol = np.linalg.cholesky(K)
        residual = self._y - cfg.prior_mean
        L_inv_r = np.linalg.solve(self._chol, residual)
        self._alpha = np.linalg.solve(self._chol.T, L_inv_r)
