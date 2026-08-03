"""
Cash-balance (2-parameter) configuration.

Two investor classes:
  * Individual   — frequent, small deposits/withdrawals (Brownian on aum_ind).
  * Institutional — rare, large deposits/withdrawals (Poisson × lognormal on aum_inst).

Policy: two-vector θ = (θ₁, θ₂). Target cash at end of day is
    target = θ₁ · aum_ind + θ₂ · aum_inst.

Shortfalls attributed by cause: institutional shortfalls force immediate
liquidation and incur the high borrow-penalty rate; individual shortfalls
are deferred and incur a much lower rate.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # --- Fund ---
    initial_aum: float = 1_000_000.0
    initial_aum_ind_fraction: float = 0.5   # fraction of AUM from individuals at t=0

    # --- Individual investors: GBM on aum_ind (small, frequent) ---
    mu_ind_annual: float = 0.00
    sigma_ind_annual: float = 0.03

    # --- Institutional investors: Poisson × lognormal on aum_inst (rare, large) ---
    # Jumps are sized relative to aum_inst (roughly half of total AUM by default),
    # so scale mean_log accordingly to produce visible shortfalls in low-buffer
    # regimes. median jump = exp(-2.5) ≈ 8% of aum_inst.
    jump_rate_inst_annual: float = 12.0
    jump_mean_log_inst: float = -2.5        # median jump ~8% of aum_inst
    jump_std_log_inst: float = 0.6
    jump_inflow_prob_inst: float = 0.5

    # --- Market regime (same 2-state HMM as 1D version) ---
    stationary: bool = False
    mu_regime_annual: tuple[float, float] = (0.001, -0.001)
    regime_switch_prob: float = 0.01
    initial_regime: int = 0

    # --- Rates ---
    r_market_annual: float = 0.10           # market return on invested (per year)
    r_cash_annual: float = 0.04             # return on cash reserves (per year)
    r_borrow_inst_annual: float = 0.02      # forced-liquidation fee (institutional)
    r_borrow_ind_annual: float = 0.005      # small goodwill-cost for deferred individual payment

    # --- Opportunity-cost formulation ---
    # True  → drag on ALL cash held × (r_market - r_cash)
    # False → drag only on cash above target
    opp_cost_on_total_cash: bool = True

    # --- Rebalancing ---
    rebalance_speed: float = 1.0

    # --- θ domain (2-vector) ---
    # theta[0] = individual buffer fraction (applies to aum_ind)
    # theta[1] = institutional buffer fraction (applies to aum_inst)
    impparam_min: tuple[float, float] = (0.01, 0.01)
    impparam_max: tuple[float, float] = (0.20, 0.40)

    # --- Calendar ---
    trading_days_per_year: int = 210
