# Modeling Decisions

Every design choice that isn't obvious from first principles is recorded here,
with rationale and the alternatives that were considered.  Future phases should
update this file rather than silently diverging from it.

---

## D1 — Opportunity cost formulation

**The conflict:** The original project spec says "charge opportunity cost only
on cash ABOVE C*," but C* is the rebalancing target, so the manager
deliberately steers cash *to* C*, not just above it.  The two formulations
differ in how they compute the drag component and produce different U-shapes.

| Option | Formula | Notes |
|--------|---------|-------|
| A (default) | `drag_rate × max(cash, 0)` | All cash earns below market.  Clean economic meaning: every dollar in cash is a dollar not invested. |
| B | `drag_rate × max(cash − C*·AUM, 0)` | Only the overshoot above target is penalized.  Implies C* buffer is "free," which breaks the U-shape (the marginal cost of *holding* C* is zero). |

**Decision:** Option A is the default (`opp_cost_on_total_cash=True`).
Rationale: the whole point of the simulator is to surface that holding *any*
cash has an opportunity cost, and the manager must balance that against
shortfall risk.  If only the excess is charged, a very small C* has
essentially zero drag, which is economically wrong.

Option B is retained as a configurable flag for sensitivity analysis.

---

## D2 — Rebalancing rule

**Options considered:**
1. Full end-of-day rebalance (`rebalance_speed=1.0`): cash is moved to exactly
   `C* × AUM` each day.  Simple, tractable, easiest to test.
2. Partial adjustment (`0 < speed < 1`): each day close a fraction of the gap.
   More realistic (large block trades take time), but adds a free parameter.

**Decision:** Default is `rebalance_speed=1.0` (full daily rebalance).
The `rebalance_speed` parameter is wired through the config and dynamics so
Phase 2 can explore partial adjustment without structural rework.

One constraint is enforced: cash cannot be refilled beyond the invested pool
(no leverage). If invested assets would go negative, the transfer is capped.

---

## D3 — AUM dynamics

**Question:** Do invested assets earn `r_market` each day, making AUM grow
over the horizon?

**Decision:** Yes. Invested assets compound at `r_market_annual / 252` per day.
Rationale: the fund exists to generate returns; a static-pool model would
misstate both the opportunity cost (which scales with AUM) and the rebalancing
targets.  The accounting test verifies this is consistent.

Consequence: the absolute target `C* × AUM` drifts upward with time, which
is the correct economic behavior.

---

## D4 — Seed derivation

`np.random.SeedSequence(session_seed).spawn(experiment_index + 1)[experiment_index]`

This uses NumPy's SeedSequence mechanism to produce independent, non-overlapping
sub-streams.  Properties:
- Same `(session_seed, experiment_index)` → identical sub-stream (reproducibility).
- Different `experiment_index` → statistically independent paths (no overlap
  in the underlying bit streams for any reasonable number of experiments).
- No collision risk at large `experiment_index`.

All randomness within one experiment (retail GBM, jump counts, jump sizes,
jump directions, regime transitions) flows from this single child generator.

---

## D5 — Retail flow model

Net retail flow is modeled as a GBM increment on the fraction-of-AUM scale:

`flow_t = AUM_t × (μ_eff dt + σ √dt × Z)`, Z ~ N(0,1)

This is a flow process, not an AUM process.  Flows are signed (inflow = positive,
outflow = negative) and added directly to cash.

The effective drift μ_eff = μ_net + μ_regime (per-day), where μ_regime is
drawn from the active HMM state.

---

## D6 — Institutional jump model

Jump times: Poisson process with annual rate `jump_rate_annual`.  In any given
day, the number of jumps is `Poisson(jump_rate_annual / 252)`.

Jump size: lognormal — `|jump| / AUM ~ LogNormal(jump_mean_log, jump_std_log)`.
Default `jump_mean_log=-4.5` gives a median jump of `exp(-4.5) ≈ 1.1%` of AUM.

Direction: Bernoulli with `P(inflow) = jump_inflow_prob` (default 0.5).

Multiple jumps in one day are possible and their flows are summed before
hitting cash.

---

## D7 — Cash negativity and shortfall

Cash is *allowed* to go negative intra-day after flows arrive.  The rebalancer
then tries to refill from invested assets.  After rebalancing, if cash is still
negative (because invested assets were insufficient), the shortfall penalty
accrues on the magnitude of the negative balance.

This means `shortfall_cost > 0` implies the fund literally ran out of cash
to cover redemptions, which is the intended economic scenario.

---

## D8 — Shortfall cost: proportional penalty, not daily interest

**The problem discovered during testing:** With daily full rebalancing, any
cash shortfall is covered within one day by liquidating invested assets.  The
naive formula `r_borrow_daily × shortfall` (annual rate ÷ 252) produces costs
on the order of pennies per event, while opportunity cost is hundreds of
dollars per year — the U-shape cannot emerge.

**Economic framing:** When redemptions exceed cash, the manager must liquidate
invested assets in a hurry.  The true cost is a *friction cost* (transaction
costs, market impact, reputational harm from forced sales), not a single day's
interest on a credit line.

**Decision:** `shortfall_cost = r_borrow × Σ shortfall_t` — a proportional
penalty per dollar of forced liquidation, with no time-unit scaling.
`r_borrow` should be read as a "friction penalty rate" (dimensionless fraction
per dollar), not an annual interest rate.  With `r_borrow = 0.10`, a $10,000
shortfall costs $1,000 immediately — comparable in magnitude to the
continuous opportunity cost, so the U-shape emerges naturally.

**Implementation note:** `shortfall_series[t]` is the pre-rebalance shortfall
(how far cash went negative before the rebalancer restored it).  It is stored
in `SimResult` so consumers can inspect individual-day friction events.

**Swappability:** A future variant could restore the time-based formula by
setting `r_borrow_annual` very high (e.g., 50× the market rate) or by adding a
`shortfall_is_rate: bool` flag to `SimConfig`.

---

## Out-of-scope (Phase 1)

The following items were intentionally deferred:
- FastAPI HTTP adapter (shell/)
- React / JS frontend (shell/)
- Belief model over F(C*) (policy/)
- Information-Gain and Knowledge-Gradient acquisition policies (policy/)
- Session history / persistence
- Charts and visualization
