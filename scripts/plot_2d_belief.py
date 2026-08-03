"""
Run KG on the 2-parameter cash-balance app for N steps and produce two
6-panel figures:

  docs/2d_belief_evolution.png  — GP posterior mean surface (Z = mean cost).
  docs/2d_kg_evolution.png      — KG(θ) surface (Z = value of information).

Both figures share the flatter perspective Warren asked for.  The KG plot
uses a fixed Z-range across all 6 snapshots so the "surface drops as we
learn" phenomenon is visually obvious.

Usage:
    python scripts/plot_2d_belief.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from apps.cash_balance_2d import simulate, SimConfig
from policy.belief import BeliefConfig
from policy.acquire import AcquisitionConfig, KGPolicy, kg_analytic_correlated_at
from policy.session import Session, SessionConfig


N_STEPS = 6
SEED = 42

# --- Configs ---
sim_cfg    = SimConfig(stationary=True)
belief_cfg = BeliefConfig(
    length_scale=(0.03, 0.06),
    signal_std=3000.0,
    noise_std=800.0,
    prior_mean=3000.0,
)
acq_cfg = AcquisitionConfig(
    impparam_min=sim_cfg.impparam_min,   # (0.01, 0.01)
    impparam_max=sim_cfg.impparam_max,   # (0.20, 0.40)
    grid_size=25,                        # 25 * 25 = 625 candidates for KG policy
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

# --- Dense grid for BELIEF surface (40x40) ---
belief_axis1 = np.linspace(acq_cfg.impparam_min[0], acq_cfg.impparam_max[0], 40)
belief_axis2 = np.linspace(acq_cfg.impparam_min[1], acq_cfg.impparam_max[1], 40)
B1, B2 = np.meshgrid(belief_axis1, belief_axis2, indexing="ij")
belief_grid = np.stack([B1.ravel(), B2.ravel()], axis=-1)   # (1600, 2)

# --- Coarser grid for KG surface (25x25 — KG per candidate is O(m^2 log m)) ---
kg_axis1 = np.linspace(acq_cfg.impparam_min[0], acq_cfg.impparam_max[0], 25)
kg_axis2 = np.linspace(acq_cfg.impparam_min[1], acq_cfg.impparam_max[1], 25)
K1, K2 = np.meshgrid(kg_axis1, kg_axis2, indexing="ij")
kg_grid = np.stack([K1.ravel(), K2.ravel()], axis=-1)       # (625, 2)

# --- Collect snapshots ---
belief_snaps: list[tuple[str, np.ndarray, list]] = []   # (title, mean_grid, obs)
kg_snaps:     list[tuple[str, np.ndarray]] = []          # (title, kg_grid)

def snapshot(title: str):
    mu, _ = session.belief.posterior(belief_grid)
    obs = [(np.asarray(t), c) for t, c in session.history]
    belief_snaps.append((title, mu.reshape(B1.shape), obs))
    kg_vals = kg_analytic_correlated_at(session.belief, kg_grid, kg_grid)
    kg_snaps.append((title, kg_vals.reshape(K1.shape)))

# Prior (no observations yet)
snapshot("Prior (0 obs)")

for step_idx in range(N_STEPS):
    result = session.step()
    theta_now = np.asarray(session.history[-1][0])
    cost_now  = session.history[-1][1]
    title = f"After step {step_idx+1}   θ=({theta_now[0]:.3f}, {theta_now[1]:.3f})   cost=${cost_now:.0f}"
    snapshot(title)


# ============================================================================
# Common plotting helpers
# ============================================================================

# Flatter camera angle — matches Warren's reference sample.
ELEV = 18
AZIM = -60
BOX_ASPECT = (1, 1, 0.55)   # squish the Z axis a bit


def make_surface_fig(snaps, x_mesh, y_mesh, z_label, cmap, fixed_z=None,
                     draw_obs=False):
    """
    snaps    : list of either (title, mean_grid, obs) or (title, mean_grid).
    fixed_z  : if not None, (zmin, zmax) applied to every subplot so we can
               compare heights across snapshots.
    draw_obs : whether the third tuple element (obs) should be scattered on
               top of each surface.
    """
    fig = plt.figure(figsize=(16, 9))
    n = min(6, len(snaps))
    for i in range(n):
        entry = snaps[i]
        if draw_obs:
            title, Z, obs = entry
        else:
            title, Z = entry[:2]
            obs = None

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        try:
            ax.set_box_aspect(BOX_ASPECT)
        except Exception:
            pass

        vmin, vmax = fixed_z if fixed_z is not None else (Z.min(), Z.max())
        surf = ax.plot_surface(x_mesh, y_mesh, Z, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               alpha=0.9, edgecolor="none",
                               antialiased=True, rcount=x_mesh.shape[0],
                               ccount=x_mesh.shape[1])
        if fixed_z is not None:
            ax.set_zlim(fixed_z)
        if draw_obs and obs:
            xs = [o[0][0] for o in obs]
            ys = [o[0][1] for o in obs]
            zs = [o[1]     for o in obs]
            ax.scatter(xs, ys, zs, color="red", s=45, depthshade=True,
                       edgecolor="white", linewidth=0.8)
        ax.set_xlabel(r"$\theta_1$ (individual buffer)", fontsize=9)
        ax.set_ylabel(r"$\theta_2$ (institutional buffer)", fontsize=9)
        ax.set_zlabel(z_label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.tick_params(labelsize=7)
    return fig


# ============================================================================
# Belief-surface figure
# ============================================================================
fig1 = make_surface_fig(
    belief_snaps, B1, B2,
    z_label="Posterior mean cost",
    cmap="viridis",
    fixed_z=None,               # per-panel autoscale (belief scale changes a lot)
    draw_obs=True,
)
fig1.suptitle("2-D GP posterior belief on cash_balance_2d — KG policy, "
              f"seed={SEED}, {N_STEPS} steps",
              fontsize=13, y=0.995)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
out1 = Path(__file__).parent.parent / "docs" / "2d_belief_evolution.png"
fig1.savefig(out1, dpi=110, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".pdf"), bbox_inches="tight")
print(f"Wrote {out1}")


# ============================================================================
# KG-surface figure — fixed Z across snapshots so the "surface drops" story reads.
# ============================================================================
all_kg = np.concatenate([kg.ravel() for _, kg in kg_snaps])
kg_zmin = 0.0
kg_zmax = float(all_kg.max()) * 1.05  # slight headroom

fig2 = make_surface_fig(
    kg_snaps, K1, K2,
    z_label="KG(θ)  (expected cost reduction)",
    cmap="jet",
    fixed_z=(kg_zmin, kg_zmax),
    draw_obs=False,
)
fig2.suptitle("Knowledge gradient surface on cash_balance_2d — "
              f"KG policy, seed={SEED}, {N_STEPS} steps (Z-axis fixed across panels)",
              fontsize=13, y=0.995)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
out2 = Path(__file__).parent.parent / "docs" / "2d_kg_evolution.png"
fig2.savefig(out2, dpi=110, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
print(f"Wrote {out2}")


# ============================================================================
# Observation trajectory (for reference)
# ============================================================================
print("\nObservation trajectory (theta, cost):")
for i, (t, c) in enumerate(session.history):
    tt = np.asarray(t)
    print(f"  step {i+1}: theta=({tt[0]:.3f}, {tt[1]:.3f})   cost=${c:>6.0f}")
best = session.best_impparam()
print(f"\nBest theta after {N_STEPS} steps: ({best[0]:.3f}, {best[1]:.3f})")
print(f"KG range over snapshots: [{kg_zmin:.1f}, {kg_zmax:.1f}]")
