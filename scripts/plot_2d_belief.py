"""
Run KG on the 2-parameter cash-balance app for N steps and produce a
6-panel figure showing the GP posterior mean surface after each update.

Usage:
    python scripts/plot_2d_belief.py

Saves to docs/2d_belief_evolution.png (and .pdf).
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure the repo root is importable no matter where you run this script from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from apps.cash_balance_2d import simulate, SimConfig
from policy.belief import BeliefConfig, BeliefModel
from policy.acquire import AcquisitionConfig, KGPolicy, _make_grid
from policy.session import Session, SessionConfig


N_STEPS = 6
SEED = 42

# --- Configs ---
sim_cfg    = SimConfig(stationary=True)
belief_cfg = BeliefConfig(
    length_scale=(0.03, 0.06),   # per-dim: individual buffer varies slower than institutional
    signal_std=3000.0,
    noise_std=800.0,
    prior_mean=3000.0,
)
acq_cfg = AcquisitionConfig(
    impparam_min=sim_cfg.impparam_min,   # (0.01, 0.01)
    impparam_max=sim_cfg.impparam_max,   # (0.20, 0.40)
    grid_size=25,                        # 25 * 25 = 625 candidates
    z_alpha=0.0,
)
ses_cfg = SessionConfig(horizon_weeks=26, best_impparam_grid=40)

# --- Session with KG on cash_balance_2d ---
policy = KGPolicy(acq_cfg)
session = Session(
    sim_config=sim_cfg,
    belief_config=belief_cfg,
    acq_config=acq_cfg,
    session_config=ses_cfg,
    policy=policy,
    session_seed=SEED,
    simulate_fn=simulate,
)

# --- Dense grid for plotting (finer than the acquisition grid) ---
plot_axis1 = np.linspace(acq_cfg.impparam_min[0], acq_cfg.impparam_max[0], 40)
plot_axis2 = np.linspace(acq_cfg.impparam_min[1], acq_cfg.impparam_max[1], 40)
G1, G2 = np.meshgrid(plot_axis1, plot_axis2, indexing="ij")
plot_grid = np.stack([G1.ravel(), G2.ravel()], axis=-1)   # (1600, 2)

# --- Snapshots of belief mean after each update ---
snapshots = []      # list of (title, mean_grid, obs_history_up_to_here)

# Prior (no observations yet)
mu_prior, _ = session.belief.posterior(plot_grid)
snapshots.append(("Prior (0 obs)", mu_prior.reshape(G1.shape), []))

for step_idx in range(N_STEPS):
    result = session.step()
    theta_now = np.asarray(session.history[-1][0])
    cost_now  = session.history[-1][1]
    mu, _ = session.belief.posterior(plot_grid)
    obs = [(np.asarray(t), c) for t, c in session.history]
    title = f"After step {step_idx+1}   θ={tuple(f'{t:.3f}' for t in theta_now)}   cost=${cost_now:.0f}"
    snapshots.append((title, mu.reshape(G1.shape), obs))

# --- Plot: 6 subplots in a 2×3 grid; each is a 3D surface ---
fig = plt.figure(figsize=(15, 9))
n = min(6, len(snapshots))
for i in range(n):
    title, mean_grid, obs = snapshots[i]
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    surf = ax.plot_surface(G1, G2, mean_grid, cmap="viridis",
                           alpha=0.85, edgecolor="none", antialiased=True)
    # Overlay observation points as red dots at their observed (noisy) cost.
    if obs:
        xs = [o[0][0] for o in obs]
        ys = [o[0][1] for o in obs]
        zs = [o[1]     for o in obs]
        ax.scatter(xs, ys, zs, color="red", s=45, depthshade=True,
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel(r"$\theta_1$ (individual buffer)", fontsize=9)
    ax.set_ylabel(r"$\theta_2$ (institutional buffer)", fontsize=9)
    ax.set_zlabel("Posterior mean cost", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-55)
    ax.tick_params(labelsize=7)

fig.suptitle("2-D GP posterior belief on cash_balance_2d — KG policy, "
             f"seed={SEED}, {N_STEPS} steps",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_png = Path(__file__).parent.parent / "docs" / "2d_belief_evolution.png"
out_pdf = out_png.with_suffix(".pdf")
fig.savefig(out_png, dpi=110, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")

# Print observation trajectory to console for reference
print("\nObservation trajectory (theta, cost):")
for i, (t, c) in enumerate(session.history):
    tt = np.asarray(t)
    print(f"  step {i+1}: theta=({tt[0]:.3f}, {tt[1]:.3f})   cost=${c:>6.0f}")
best = session.best_impparam()
print(f"\nBest theta after {N_STEPS} steps: ({best[0]:.3f}, {best[1]:.3f})")
