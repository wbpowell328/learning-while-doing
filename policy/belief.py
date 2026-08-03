"""
Gaussian Process belief model over F(θ) = E[total_cost | θ].

Kernel: squared-exponential (RBF)
    k(x, x') = signal_std² × exp(-0.5 × (x - x')² / length_scale²)

Observations y_i = F(C*_i) + ε_i, ε_i ~ N(0, noise_std²).

The posterior is computed from scratch on each query after a new observation
(lazy Cholesky cache, invalidated on update).  This is correct and simple;
incremental Cholesky updates can be added if n_observations grows large.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeliefConfig:
    # RBF kernel hyperparameters
    length_scale: float = 0.04    # length scale in θ units; θ ∈ [~0.01, ~0.20]
    signal_std: float = 5_000.0   # prior amplitude: std of F(θ) in cost units
    noise_std: float = 3_000.0    # per-observation noise std (single-run variance)
    prior_mean: float = 5_000.0   # constant prior mean for F(θ)

    # Numerical stability: added to kernel diagonal before Cholesky
    jitter: float = 1e-6


class BeliefModel:
    """
    GP surrogate for the expected total cost F(θ).

    Usage:
        model = BeliefModel()           # or BeliefModel(BeliefConfig(...))
        model.update(c_star, cost)      # add one observation
        mean, std = model.posterior(grid)  # query the posterior
    """

    def __init__(self, config: BeliefConfig | None = None) -> None:
        self.config: BeliefConfig = config if config is not None else BeliefConfig()
        self._x_obs: list[float] = []
        self._y_obs: list[float] = []
        self._chol: np.ndarray | None = None   # lower-triangular Cholesky of K
        self._alpha: np.ndarray | None = None  # K^{-1} (y - prior_mean)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, c_star: float, observed_cost: float) -> None:
        """Append one (θ, cost) observation and invalidate the cached decomposition."""
        self._x_obs.append(float(c_star))
        self._y_obs.append(float(observed_cost))
        self._chol = None
        self._alpha = None

    def posterior(self, c_stars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) at each query point in c_stars.

        std is the posterior standard deviation, clipped to zero to absorb
        small negative values from floating-point arithmetic.
        """
        c_stars = np.asarray(c_stars, dtype=float)
        cfg = self.config

        if not self._x_obs:
            mean = np.full_like(c_stars, cfg.prior_mean)
            std = np.full_like(c_stars, cfg.signal_std)
            return mean, std

        self._build_cache()

        X = np.array(self._x_obs)

        # k(c_stars, X): shape (m, n)
        K_star_X = self._rbf(c_stars[:, None], X[None, :])

        # Posterior mean: m₀ + k(x*, X) α
        mean = cfg.prior_mean + K_star_X @ self._alpha  # type: ignore[operator]

        # Posterior variance: k(x*,x*) - ||L⁻¹ k(X,x*)||²
        # k(x*, x*) = signal_std² for all x* (diagonal of prior covariance)
        V = np.linalg.solve(self._chol, K_star_X.T)  # (n, m)
        post_var = cfg.signal_std ** 2 - np.sum(V ** 2, axis=0)
        std = np.sqrt(np.maximum(post_var, 0.0))

        return mean, std

    def posterior_at(self, c_star: float) -> tuple[float, float]:
        """Convenience wrapper: return (mean, std) at a single point."""
        mean, std = self.posterior(np.array([float(c_star)]))
        return float(mean[0]), float(std[0])

    def posterior_cov_matrix(self, c_stars: np.ndarray) -> np.ndarray:
        """
        Full posterior covariance matrix at query points, shape (m, m).

        Cov_n(x_i, x_j) = k(x_i, x_j) − V_i · V_j
        where V = L⁻¹ k(X, c_stars), L = Cholesky(K(X,X) + noise·I).

        With no observations this equals the prior covariance k(c_stars, c_stars).
        """
        c_stars = np.asarray(c_stars, dtype=float)
        K_star = self._rbf(c_stars[:, None], c_stars[None, :])  # (m, m)

        if not self._x_obs:
            return K_star

        self._build_cache()

        X = np.array(self._x_obs)
        k_X_grid = self._rbf(X[:, None], c_stars[None, :])     # (n, m)
        V = np.linalg.solve(self._chol, k_X_grid)              # (n, m)
        return K_star - V.T @ V                                 # (m, m)

    @property
    def n_observations(self) -> int:
        return len(self._x_obs)

    @property
    def observations(self) -> tuple[list[float], list[float]]:
        """Return (c_stars, costs) as independent copies."""
        return list(self._x_obs), list(self._y_obs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rbf(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Squared-exponential kernel; x1 and x2 broadcast to the output shape."""
        cfg = self.config
        sq_dist = (x1 - x2) ** 2
        return cfg.signal_std ** 2 * np.exp(-0.5 * sq_dist / cfg.length_scale ** 2)

    def _build_cache(self) -> None:
        """Compute and cache the Cholesky of K(X,X) and the weight vector α."""
        if self._chol is not None:
            return

        cfg = self.config
        X = np.array(self._x_obs)
        y = np.array(self._y_obs)
        n = len(X)

        K = self._rbf(X[:, None], X[None, :])
        K += (cfg.noise_std ** 2 + cfg.jitter) * np.eye(n)

        self._chol = np.linalg.cholesky(K)

        residual = y - cfg.prior_mean
        L_inv_r = np.linalg.solve(self._chol, residual)
        self._alpha = np.linalg.solve(self._chol.T, L_inv_r)
