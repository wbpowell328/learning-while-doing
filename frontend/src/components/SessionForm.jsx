import { useState, useEffect, useRef } from 'react';
import { createSession, getReveal, deleteSession } from '../api';

// Defaults must match backend Pydantic defaults (shell/models.py) so the
// initial UI state is consistent with what the API would use for an empty
// request. All values are held as strings during typing and coerced to
// numbers on submit; onBlur canonicalizes ("050" → "50").
const ADV_DEFAULTS = {
  // Belief prior — the GP's assumptions before it sees any data.
  // Values are in the *reward-per-day* frame so the belief stays
  // coherent when Run length changes mid-session. On a ~200-day
  // batch, per-batch dollars ≈ n_days × per-day dollars:
  //   prior_mean  175   ≈ $35k per 200-day batch
  //   signal_std   15   ≈ $3k  per 200-day batch (linear in n)
  //   noise_std   100   ≈ $1.4k per 200-day batch (scales as √n)
  // Session divides observations by n_days on ingest so mixed batch
  // lengths land as comparable per-day numbers.
  length_scale:      '0.04',
  signal_std:        '15',
  noise_std:         '100',
  prior_mean:        '175',
  // Starting θ that pre-fills the ExperimentBar's "Starting point"
  // box (and thereby the initial cash-balance sample-path chart).
  // Two components for the 2-D app.
  initial_theta:     '0.10',
  initial_theta1:    '0.10',
  initial_theta2:    '0.10',
  // Ryzhov's N — total budget of experiments used by the classical
  // online KG "μ − (N−n)·KG" policy. Ignored by every other policy.
  ryzhov_budget:     '10',
  // Fund size ("AUM" = assets under management). Kept editable so
  // users can shrink the fund without also having to rescale the
  // dollar-denominated median jump — a smaller AUM at fixed jump-$
  // makes each jump a larger fraction of AUM, which shifts θ*
  // meaningfully upward.
  initial_aum:       '1000000',
  // Simulation model (1-D app only) — the underlying truth F(θ) and its per-run noise.
  // Jump size is displayed in dollars for legibility; converted to
  // jump_mean_log = ln(median$ / initial_aum) on submit. initial_aum
  // is the 1-D config default of $1,000,000; if that ever changes,
  // the conversion below has to change too.
  jump_rate_annual:  '12',
  jump_median_dollars: '11000',   // median jump = $11k on $1M AUM  ~1.1% AUM (exp(-4.5))
  jump_std_log:      '0.5',
  sigma_net_annual:  '0.02',
  r_borrow_annual:   '0.10',
  // θ search-box bounds (1-D uses first component only; 2-D uses both)
  theta1_min:        '0.01',
  theta1_max:        '0.20',
  theta2_min:        '0.01',
  theta2_max:        '0.40',
  // Objective / market rates (2-D app)
  r_market_annual:       '0.10',   // annual return on invested
  r_cash_annual:         '0.04',   // annual return on cash
  trading_days_per_year: '210',    // number of days market is open per year
  // Individual investor process (2-D app) — GBM on aum_ind
  mu_ind_annual:         '0.00',   // drift of individual net flow
  sigma_ind_annual:      '0.03',   // volatility of individual net flow
  r_borrow_ind_annual:   '0.005',  // goodwill cost per $ of deferred individual redemption
  // Institutional investor process (2-D app) — Poisson × lognormal on aum_inst
  jump_rate_inst_annual: '12',     // jumps/year
  // 2-D institutional jump median in $ (converted to jump_mean_log_inst
  // on submit using aum_inst = initial_aum × initial_aum_ind_fraction
  // complement = $500k by default). exp(-2.5) × $500k ≈ $41k.
  jump_median_dollars_inst: '41000',
  jump_std_log_inst:        '0.6',
  r_borrow_inst_annual:     '0.02',   // 2% redemption fee on forced institutional liquidation
};

// AUM used for the median-dollars ↔ jump_mean_log conversion is now
// pulled from the editable adv:initial_aum field at submit time; see
// _initial_aum below. (The hardcoded 1_000_000 / 500_000 constants
// this file used to keep are gone — dynamic value is the source of
// truth so a user's AUM edit propagates correctly.)

// Field metadata for the advanced-parameters panel
const BELIEF_FIELDS = [
  ['length_scale', 'Length scale ℓ',
    'Smoothness of prior F(θ). Larger = fewer wiggles.'],
  ['signal_std', 'Signal std σ_f',
    'Prior amplitude — how much F(θ) is expected to vary across θ (in reward $).'],
  ['noise_std', 'Noise std σ_n',
    'Per-observation reward noise the GP assumes (algorithm side).'],
  ['prior_mean', 'Prior mean m₀',
    'Typical run reward, before any observations. (Reward frame.)'],
];

const SIM_FIELDS = [
  ['jump_rate_annual', 'Jump rate (/yr)',
    'Poisson rate of large inflow/outflow shocks.'],
  ['jump_std_log', 'Jump std (log)',
    'Log-space std of jump sizes.'],
  ['sigma_net_annual', 'Base flow σ (/yr)',
    'Baseline daily-flow volatility (non-jump).'],
  ['r_borrow_annual', 'Borrow rate (/yr)',
    'Rate paid when cash goes negative — controls the low-θ cliff.'],
];

// θ search-box bounds. For a 1-D app, only theta1_min/max are used.
// For a 2-D app, all four are used.
const RANGE_FIELDS_1D = [
  ['theta1_min', 'θ min', 'Lower bound of the θ search box.'],
  ['theta1_max', 'θ max', 'Upper bound of the θ search box.'],
];
const RANGE_FIELDS_2D = [
  ['theta1_min', 'θ₁ min (individual)', 'Lower bound for individual buffer θ₁.'],
  ['theta1_max', 'θ₁ max (individual)', 'Upper bound for individual buffer θ₁.'],
  ['theta2_min', 'θ₂ min (institutional)', 'Lower bound for institutional buffer θ₂.'],
  ['theta2_max', 'θ₂ max (institutional)', 'Upper bound for institutional buffer θ₂.'],
];

// --- 2-D-only stochastic + objective parameters ---

const MARKET_FIELDS_2D = [
  ['r_market_annual',       'Market return (r_market, /yr)',
    'Annual return on invested assets. Daily rate = value / trading_days_per_year.'],
  ['r_cash_annual',         'Cash return (r_cash, /yr)',
    'Annual return on cash reserves. Drag on cash = r_market − r_cash.'],
  ['trading_days_per_year', 'Trading days per year',
    'Divide annual rates by this to get daily rates.'],
];

const IND_FIELDS_2D = [
  ['mu_ind_annual',       'Individual drift (μ_ind, /yr)',
    'Annualized mean of individual net flow / aum_ind.'],
  ['sigma_ind_annual',    'Individual volatility (σ_ind, /yr)',
    'Annualized std of individual daily flow / aum_ind.'],
  ['r_borrow_ind_annual', 'Individual deferral fee',
    'Cost per $ of deferred individual redemption (small — they can wait).'],
];

const INST_FIELDS_2D = [
  ['jump_rate_inst_annual', 'Institutional jump rate (/yr)',
    'Poisson rate of large institutional in/outflows.'],
  ['jump_mean_log_inst',    'Jump size mean (log)',
    'Mean of log(|J|/aum_inst). Median jump = exp(this).'],
  ['jump_std_log_inst',     'Jump size std (log)',
    'Log-space spread of jump sizes.'],
  ['r_borrow_inst_annual',  'Institutional redemption fee',
    'Cost per $ of forced liquidation to meet an institutional withdrawal.'],
];


// Available applications — the values here must match keys in the backend's
// apps/__init__.py REGISTRY. dim tells the frontend how to shape θ inputs
// and which visualization to show for the belief posterior.
const APPS = [
  { value: 'cash_balance',    label: 'Cash balance (1-parameter)',                dim: 1 },
  { value: 'cash_balance_2d', label: 'Cash balance (2-parameter, θ_ind, θ_inst)', dim: 2 },
];

// Row schemas for the Variable | Default | Range | Explanation table.
// Mirrors Warren's "Cash management - advanced parameters" spreadsheet.
// `source` addresses where the value lives:
//   'reportLevel', 'seed', 'mStarStr'     → direct component state
//   'adv:<key>'                            → adv[key] (the ADV_DEFAULTS map)
// Row `kind`:
//   'section'  → header row spanning all columns
//   'select'   → dropdown with `options`
//   'int'      → integer input
//   'number'   → floating-point input
//   'pair'     → two side-by-side inputs (Default cell holds both)
const ROWS_1D = [
  { kind: 'section', label: 'Administrative' },
  { kind: 'select', label: 'Report level', source: 'reportLevel',
    options: [
      { value: 'basic',    label: 'Basic' },
      { value: 'advanced', label: 'Advanced' },
    ],
    range: 'Basic, advanced',
    desc: '"Basic" shows just the reports needed to play the game. "Advanced" adds diagnostic panels — the KG(x;m) S-curve chart, for one — for understanding model behavior.' },

  { kind: 'section', label: 'Parameters governing search policies' },
  { kind: 'int', label: 'Lookahead horizon for knowledge gradient (ρˡᵏʰᵈ)',
    source: 'mStarStr', min: 1, max: 100,
    range: '[1–100]',
    desc: 'If =1, the KG computes the value of a single experiment for the decision we make now. If >1, the KG assumes we perform ρˡᵏʰᵈ replications, reducing the noise of the observation.' },
  { kind: 'number', label: 'Search box bound — min',
    source: 'adv:theta1_min', step: 'any', min: 0,
    range: '',
    desc: 'Lower limit for the fraction of assets under management (AUM) held in cash.' },
  { kind: 'number', label: 'Search box bound — max',
    source: 'adv:theta1_max', step: 'any', min: 0,
    range: '',
    desc: 'Upper limit for the fraction of assets under management (AUM) held in cash.' },
  { kind: 'number', label: 'Initial value of θ',
    source: 'adv:initial_theta', step: 'any', min: 0,
    range: 'inside the θ box',
    desc: 'Starting cash-buffer fraction that pre-fills the "Starting point" box on the game\'s control bar. Also seeds the initial cash-balance sample-path chart.' },
  { kind: 'int', label: 'Ryzhov budget N',
    source: 'adv:ryzhov_budget', min: 1, max: 500,
    range: '1 – 500',
    desc: 'Total experiment budget used by the classical Ryzhov online-KG policy (μ − (N−n)·KG). The exploration bonus is largest at n=0 and decays to 0 as n approaches N. Ignored by every other policy.' },

  { kind: 'section', label: 'Parameters controlling belief about the profit function (all per day)' },
  { kind: 'number', label: 'Length scale (ℓ)',
    source: 'adv:length_scale', step: 'any', min: 0,
    range: '0.01 – 10',
    desc: 'Smoothness of the belief about the profit function. Larger produces a smoother graph — testing one value of θ teaches us more about the entire function.' },
  { kind: 'number', label: 'Initial estimate of profit function ($/day)',
    source: 'adv:prior_mean', step: 'any',
    range: '$50 – $2,000 / day',
    desc: 'Expected daily reward before any observations. A 200-day batch would then earn ≈ 200 × this value. Given per-day so a change to Run length in the control bar rescales correctly.' },
  { kind: 'number', label: 'Variation of the profit function (σ_f, $/day)',
    source: 'adv:signal_std', step: 'any', min: 0,
    range: '$1 – $500 / day',
    desc: 'Prior belief about how much the daily profit function F(θ) might vary. The GP treats F(θ) at any θ as normally distributed around the initial estimate with this standard deviation. Larger σ_f → the belief is open to bigger swings across θ; smaller → the belief is confident F is close to the initial estimate everywhere.' },
  { kind: 'number', label: 'Std. deviation of a daily observation (σ_n, $/day)',
    source: 'adv:noise_std', step: 'any', min: 0,
    range: '$10 – $1,000 / day',
    desc: 'Belief about per-day observation noise — the GP\'s assumption for how much a single day\'s reward jitters around the true F(θ). This is the algorithm\'s input, not the actual noise the sim generates; belief-vs-truth mismatch here is itself pedagogically interesting.' },

  { kind: 'section', label: 'Model of deposits and redemptions' },
  { kind: 'number', label: 'Initial AUM ($)',
    source: 'adv:initial_aum', step: 'any', min: 1000,
    range: '$100k – $10M+',
    desc: 'Starting size of the fund. Doesn\'t affect θ* directly, but a smaller AUM at a fixed median-jump-in-dollars makes each jump a larger fraction of AUM, so θ* shifts up. Convenient knob for shaping where the optimum sits.' },
  { kind: 'number', label: 'Return on invested (r_market, /yr)',
    source: 'adv:r_market_annual', step: 'any', min: 0,
    range: '0.05 – 0.20',
    desc: 'Annualized return on money invested in the market. The gap (r_market − r_cash) is the opportunity cost of every dollar you hold as cash — that\'s what steepens the reward-vs-θ curve on the "too much cash" side.' },
  { kind: 'number', label: 'Return on cash (r_cash, /yr)',
    source: 'adv:r_cash_annual', step: 'any', min: 0,
    range: '0 – 0.06',
    desc: 'Annualized yield on cash reserves. Smaller r_cash (or a larger r_market − r_cash spread) makes holding excess cash more costly and pulls θ* down.' },
  { kind: 'number', label: 'Annual volatility of daily deposits / redemptions (fraction of AUM / yr)',
    source: 'adv:sigma_net_annual', step: 'any', min: 0,
    range: '0.01 – 0.20',
    desc: 'Annualized standard deviation of the retail flow, expressed as a fraction of AUM. The simulator converts to a per-day dollar magnitude via σ_daily × AUM = (σ_annual / √252) × AUM. Default 0.02 → daily flow std ≈ 0.126 % × AUM per day ($1,260/day on a $1M fund, $126/day on $100k).' },
  { kind: 'number', label: 'Borrowing cost',
    source: 'adv:r_borrow_annual', step: 'any', min: 0,
    range: '0.02 – 0.30',
    desc: 'Dollars lost per dollar of shortfall.' },
  { kind: 'number', label: 'Jump rate (per year)',
    source: 'adv:jump_rate_annual', step: 'any', min: 0,
    range: '0 – 120',
    desc: 'Average rate of large inflow / outflow shocks per year.' },
  { kind: 'number', label: 'Median jump size ($)',
    source: 'adv:jump_median_dollars', step: 'any', min: 1,
    range: '$1k – $500k',
    desc: 'Dollar size of the "typical" (50th-percentile) jump. On a $1M fund, $11k is ~1.1% of AUM; $50k is ~5%. This is what makes low-θ shortfalls likely — bigger median jumps mean bigger cash buffers are worth holding.' },
  { kind: 'number', label: 'Jump-size spread (log std)',
    source: 'adv:jump_std_log', step: 'any', min: 0,
    range: '0.1 – 2',
    desc: 'Spread of jump sizes on a log scale. sd = 0.5 → the 84th-percentile jump is ≈ 1.65 × median; sd = 1.5 → ≈ 4.5 × median (fat tail).' },

  { kind: 'section', label: 'Session' },
  { kind: 'int', label: 'Random number seed', source: 'seed', min: 0, max: 9999,
    range: '',
    desc: 'Seed governing the generation of deposits and redemptions.' },
];

const ROWS_2D = [
  { kind: 'section', label: 'Administrative' },
  { kind: 'select', label: 'Report level', source: 'reportLevel',
    options: [
      { value: 'basic',    label: 'Basic' },
      { value: 'advanced', label: 'Advanced' },
    ],
    range: 'Basic, advanced',
    desc: '"Basic" shows just the reports needed to play the game. "Advanced" adds diagnostic panels for understanding model behavior.' },

  { kind: 'section', label: 'Parameters governing search policies' },
  { kind: 'int', label: 'Lookahead horizon for knowledge gradient (ρˡᵏʰᵈ)',
    source: 'mStarStr', min: 1, max: 100,
    range: '[1–100]',
    desc: 'If =1, the KG computes the value of a single experiment. If >1, the KG assumes we perform ρˡᵏʰᵈ replications, reducing the noise of the observation.' },
  { kind: 'pair', label: 'Search box bounds — retail investor (θ₁)',
    sources: ['adv:theta1_min', 'adv:theta1_max'],
    step: 'any', min: 0,
    range: '[0.01 – 0.20]',
    desc: 'Min and max for the fraction of individual (retail) AUM held in cash.' },
  { kind: 'pair', label: 'Search box bounds — institutional investor (θ₂)',
    sources: ['adv:theta2_min', 'adv:theta2_max'],
    step: 'any', min: 0,
    range: '[0.01 – 0.40]',
    desc: 'Min and max for the fraction of institutional AUM held in cash.' },
  { kind: 'number', label: 'Initial value of θ₁ (retail)',
    source: 'adv:initial_theta1', step: 'any', min: 0,
    range: 'inside θ₁ box',
    desc: 'Starting individual-investor cash-buffer fraction. Pre-fills the θ₁ box on the control bar and seeds the cash-balance sample chart.' },
  { kind: 'number', label: 'Initial value of θ₂ (institutional)',
    source: 'adv:initial_theta2', step: 'any', min: 0,
    range: 'inside θ₂ box',
    desc: 'Starting institutional cash-buffer fraction. Pre-fills the θ₂ box on the control bar.' },
  { kind: 'int', label: 'Ryzhov budget N',
    source: 'adv:ryzhov_budget', min: 1, max: 500,
    range: '1 – 500',
    desc: 'Total experiment budget used by the classical Ryzhov online-KG policy (μ − (N−n)·KG). Ignored by every other policy.' },

  { kind: 'section', label: 'Parameters controlling belief about the profit function (all per day)' },
  { kind: 'number', label: 'Length scale (ℓ)',
    source: 'adv:length_scale', step: 'any', min: 0,
    range: '0.01 – 10',
    desc: 'Smoothness of the belief about the profit function. Larger produces a smoother graph.' },
  { kind: 'number', label: 'Initial estimate of profit function ($/day)',
    source: 'adv:prior_mean', step: 'any',
    range: '$50 – $2,000 / day',
    desc: 'Expected daily reward before any observations. A 200-day batch would then earn ≈ 200 × this value. Given per-day so a change to Run length in the control bar rescales correctly.' },
  { kind: 'number', label: 'Variation of the profit function (σ_f, $/day)',
    source: 'adv:signal_std', step: 'any', min: 0,
    range: '$1 – $500 / day',
    desc: 'Prior belief about how much the daily profit function F(θ) might vary. The GP treats F(θ) at any θ as normally distributed around the initial estimate with this standard deviation. Larger σ_f → the belief is open to bigger swings across θ; smaller → the belief is confident F is close to the initial estimate everywhere.' },
  { kind: 'number', label: 'Std. deviation of a daily observation (σ_n, $/day)',
    source: 'adv:noise_std', step: 'any', min: 0,
    range: '$10 – $1,000 / day',
    desc: 'Belief about per-day observation noise — the GP\'s assumption for how much a single day\'s reward jitters around the true F(θ). This is the algorithm\'s input, not the actual noise the sim generates; belief-vs-truth mismatch here is itself pedagogically interesting.' },

  { kind: 'section', label: 'Model of deposits and redemptions — individual investors' },
  { kind: 'number', label: 'Initial AUM ($)',
    source: 'adv:initial_aum', step: 'any', min: 1000,
    range: '$100k – $10M+',
    desc: 'Starting size of the fund. Split 50/50 between individual and institutional by default. Smaller AUM at fixed median-jump-in-dollars makes jumps a larger fraction of AUM, shifting θ* up.' },
  { kind: 'number', label: 'Annual net inflow',
    source: 'adv:mu_ind_annual', step: 'any',
    range: '',
    desc: 'Long-run average net flow (deposits − withdrawals). 0 = steady state (inflows ≈ outflows on average).' },
  { kind: 'number', label: 'Volatility',
    source: 'adv:sigma_ind_annual', step: 'any', min: 0,
    range: '0.01 – 0.10',
    desc: 'Annualized standard deviation of daily individual flows / AUM.' },
  { kind: 'number', label: 'Deferral fee',
    source: 'adv:r_borrow_ind_annual', step: 'any', min: 0,
    range: '0.001 – 0.05',
    desc: 'Cost per $ of deferred individual redemption — small, because individual investors can wait.' },

  { kind: 'section', label: 'Model of deposits and redemptions — institutional investors' },
  { kind: 'number', label: 'Jump rate (per year)',
    source: 'adv:jump_rate_inst_annual', step: 'any', min: 0,
    range: '1 – 120',
    desc: 'Rate of large institutional deposits or withdrawals per year.' },
  { kind: 'number', label: 'Median jump size ($)',
    source: 'adv:jump_median_dollars_inst', step: 'any', min: 1,
    range: '$1k – $250k',
    desc: 'Dollar size of the "typical" (50th-percentile) institutional jump. On $500k institutional AUM, $41k is ~8%.' },
  { kind: 'number', label: 'Jump-size spread (log std)',
    source: 'adv:jump_std_log_inst', step: 'any', min: 0,
    range: '0.1 – 2',
    desc: 'Spread on a log scale. sd = 0.6 → the 84th-percentile jump ≈ 1.8 × median.' },
  { kind: 'number', label: 'Institutional redemption fee',
    source: 'adv:r_borrow_inst_annual', step: 'any', min: 0,
    range: '0.01 – 0.20',
    desc: 'Cost per $ of forced liquidation to meet an institutional withdrawal.' },

  { kind: 'section', label: 'Session' },
  { kind: 'int', label: 'Random number seed', source: 'seed', min: 0, max: 9999,
    range: '',
    desc: 'Seed governing the generation of deposits and redemptions.' },
];

// Advanced-parameters are persisted in localStorage keyed by this string
// so the "Save and exit" flow is meaningful — the user's edits survive
// leaving the page and re-launching the game from the landing page.
// Bump the suffix if the shape of the saved blob ever changes.
// Bumped v1 → v2 when belief params flipped to per-day units. Values
// saved under the old key are auto-migrated below (once) and the old
// key deleted so the migration doesn't re-run.
const OLD_ADVANCED_STORAGE_KEY = 'lwd_advanced_v1';
const ADVANCED_STORAGE_KEY     = 'lwd_advanced_v2';
// Where "Save and exit" sends the user back to. Landing page on the
// CASTLE site — the game is embedded there behind the Play button.
const LANDING_URL = 'https://warrenpowell.org/learning-while-doing/';

function readSavedAdvanced() {
  try {
    // First: prefer the current-version blob if it exists.
    const rawNew = window.localStorage.getItem(ADVANCED_STORAGE_KEY);
    if (rawNew) {
      const parsed = JSON.parse(rawNew);
      return (parsed && typeof parsed === 'object') ? parsed : {};
    }
    // Otherwise: try the legacy per-batch-units blob and migrate.
    const rawOld = window.localStorage.getItem(OLD_ADVANCED_STORAGE_KEY);
    if (!rawOld) return {};
    const oldParsed = JSON.parse(rawOld);
    if (!oldParsed || typeof oldParsed !== 'object') return {};
    // Rescale belief params from per-200-day-batch to per-day, only
    // for values that were saved and look like the old magnitudes.
    const REF = 200;    // typical batch length used to seed the old defaults
    const advOld = (oldParsed.adv && typeof oldParsed.adv === 'object') ? oldParsed.adv : {};
    const advNew = { ...advOld };
    for (const k of ['prior_mean', 'signal_std', 'noise_std']) {
      const v = Number(advNew[k]);
      // Heuristic: anything above the new-defaults ceiling was almost
      // certainly per-batch; scale it down. noise_std uses √n (batch
      // noise = √n · per-day-noise); signal_std / prior_mean scale
      // linearly (batch = n · per-day).
      if (Number.isFinite(v) && v > 1000) {
        const scale = (k === 'noise_std') ? Math.sqrt(REF) : REF;
        advNew[k] = String(Math.round((v / scale) * 100) / 100);
      }
    }
    const migrated = { ...oldParsed, adv: advNew };
    window.localStorage.setItem(ADVANCED_STORAGE_KEY, JSON.stringify(migrated));
    window.localStorage.removeItem(OLD_ADVANCED_STORAGE_KEY);
    return migrated;
  } catch { return {}; }
}
function writeSavedAdvanced(blob) {
  try {
    window.localStorage.setItem(ADVANCED_STORAGE_KEY, JSON.stringify(blob));
  } catch { /* private mode / quota — silently accept the loss */ }
}

// ── Named-panels storage ──────────────────────────────────────────────────
// Layered ABOVE the single-blob storage: each app_name keeps a set of
// named panels + which one is currently active. Warren's 2026-08 ask.
//
// Structure:
//   {
//     "cash_balance":     { active: "Default", panels: { Default: <blob>, ... } },
//     "cash_balance_2d":  { active: "My tuning", panels: { ... } },
//     ...
//   }
// A <blob> here is the same shape writeSavedAdvanced used to persist
// (seed, stationary, mStar, zAlpha, sigmaGreedy, flowHorizon,
//  reportLevel, adv). Schema evolves by ADV_DEFAULTS merge on load, so
// panels saved before a new field was added automatically pick up the
// new field's default.
const PANELS_STORAGE_KEY = 'lwd_panels_v1';
const DEFAULT_PANEL_NAME = 'Default';

function readPanelsStore() {
  try {
    const raw = window.localStorage.getItem(PANELS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return (parsed && typeof parsed === 'object') ? parsed : {};
    }
  } catch { /* fall through to migration */ }
  return {};
}

function writePanelsStore(store) {
  try {
    window.localStorage.setItem(PANELS_STORAGE_KEY, JSON.stringify(store));
  } catch { /* private mode / quota — silently accept */ }
}

// Read the panels store, migrating from the flat lwd_advanced_v2 blob
// on first load. The old flat blob is preserved so a browser using the
// old key doesn't lose data if the user rolls back the deploy.
function initPanelsStore(appName) {
  const store = readPanelsStore();
  if (!store[appName]) {
    // First-time visit for this app — seed with a Default panel from
    // the legacy single-blob storage (if present) or empty (→ form
    // will use ADV_DEFAULTS everywhere).
    const legacy = readSavedAdvanced();
    store[appName] = {
      active: DEFAULT_PANEL_NAME,
      panels: { [DEFAULT_PANEL_NAME]: legacy && Object.keys(legacy).length ? legacy : {} },
    };
    writePanelsStore(store);
  }
  return store;
}

// Sorted panel names for the dropdown: Default first, then the rest
// in insertion order (Object.keys is insertion-preserving for string
// keys in modern JS engines).
function panelNameList(store, appName) {
  const names = Object.keys(store?.[appName]?.panels ?? {});
  const rest = names.filter(n => n !== DEFAULT_PANEL_NAME);
  return names.includes(DEFAULT_PANEL_NAME)
    ? [DEFAULT_PANEL_NAME, ...rest]
    : rest;
}

export default function SessionForm({
  onCreate, error,
  initialAppName = 'cash_balance',
  initialPolicy  = 'okg_ryzhov',
  autoSubmit     = false,   // when true (URL ?auto=1), fire submit on mount
}) {
  // App + policy come from the landing page via URL query params. The
  // dropdowns for them were moved to the landing page per Warren's spec,
  // so they aren't rendered here — but we still track the values so the
  // submit payload carries them through.
  const [appName]  = useState(initialAppName);
  const [policy]   = useState(initialPolicy);
  const [loading, setLoading] = useState(false);
  // True while precomputing the reveal (true-reward curve) at save
  // time. Used to disable the save buttons and show a "Computing…"
  // hint so the user knows why the click is pausing for a few seconds.
  const [precomputing, setPrecomputing] = useState(false);

  // Named-panels store: per-app dict of saved parameter sets
  // (Default + any user-created via "Save as new"). Loaded once on
  // mount; kept in React state so the selector re-renders when a
  // panel is added / activated.
  const [panelsStore, setPanelsStore] = useState(() => {
    const store = initPanelsStore(initialAppName);
    // Landing page can pass ?panel=<name> to force a specific panel
    // active on entry. Also honored by the in-game "top-box" panel
    // selector, which navigates here with the same query when the
    // user switches panels mid-play.
    try {
      const wanted = new URLSearchParams(window.location.search).get('panel');
      if (wanted && store[initialAppName]?.panels?.[wanted]) {
        store[initialAppName].active = wanted;
        writePanelsStore(store);
      }
    } catch (_) { /* no-op */ }
    return store;
  });
  const activePanelName = panelsStore?.[initialAppName]?.active ?? DEFAULT_PANEL_NAME;
  const savedAdv        = panelsStore?.[initialAppName]?.panels?.[activePanelName] ?? {};

  // Random seed + stationary regime — moved into the Advanced Parameters
  // panel per Warren. Weeks per run is dropped entirely (ExperimentBar's
  // "Run N days" input now sets the horizon on every iteration); we
  // still send a default horizon_weeks on session-create so the backend
  // SessionConfig has a value, but the user never sees it.
  const [seed, setSeed]             = useState(savedAdv.seed ?? 42);
  const [stationary, setStationary] = useState(savedAdv.stationary ?? true);
  const horizon = 10;   // 50 trading days — matches ExperimentBar's default N so pre-run chart scaling is right

  // Policy parameter (single field, meaning depends on the policy):
  //   KG variants → m* (days) — how many repeat experiments the policy
  //     assumes when computing KG (batch-size trick).
  //   IE → z_alpha (# std devs) — UCB coefficient (maximise frame).
  //   Random / Human → unused.
  const [mStarStr,       setMStarStr]       = useState(savedAdv.mStar       != null ? String(savedAdv.mStar)       : '1');
  const [zAlphaStr,      setZAlphaStr]      = useState(savedAdv.zAlpha      != null ? String(savedAdv.zAlpha)      : '0');
  const [sigmaGreedyStr, setSigmaGreedyStr] = useState(savedAdv.sigmaGreedy != null ? String(savedAdv.sigmaGreedy) : '0');

  // Horizon H for the "Deposits & redemptions" sample-path report on
  // the right column. Display-only — doesn't affect the belief or the
  // simulator's policy runs.
  const [flowHorizonStr, setFlowHorizonStr] = useState(savedAdv.flowHorizon != null ? String(savedAdv.flowHorizon) : '200');
  const flowHorizon = Math.max(1, Math.min(5000, Math.round(Number(flowHorizonStr) || 200)));

  // Reporting level — controls how much of the diagnostic UI is shown.
  // Basic = core charts only; Advanced also shows the KG(x;m) card.
  const [reportLevel, setReportLevel] = useState(savedAdv.reportLevel ?? 'basic');

  // Advanced parameters — kept as strings during typing.
  const [adv, setAdv] = useState({ ...ADV_DEFAULTS, ...(savedAdv.adv ?? {}) });
  const setField     = (k, v) => setAdv(prev => ({ ...prev, [k]: v }));
  // Parse an advanced field, falling back to its default ONLY when the box
  // is empty or non-numeric — never for a legitimate 0 (a `|| default`
  // here wrongly reset e.g. Return on cash = 0 back to 0.04).
  const parseAdv = (k) => {
    const s = String(adv[k] ?? '').trim();
    if (s !== '') { const n = Number(s); if (Number.isFinite(n)) return n; }
    return Number(ADV_DEFAULTS[k]);
  };
  const canonicalize = (k) => setField(k, String(parseAdv(k)));
  const numeric = (k) => parseAdv(k);

  // Selected app metadata.
  const appMeta = APPS.find(a => a.value === appName) ?? APPS[0];
  const is2D = appMeta.dim >= 2;

  // Auto-flip incompatible policy selections when app changes. 2-D
  // supports everything IE-based (including the two fixed variants
  // ie_15 and greedy) plus every KG variant plus randomized_greedy —
  // just not Manual (human), which the 2-D dropdown filters out
  // anyway.
  const policyAllowed = (p) => is2D
    ? ['random', 'ie', 'ie_15', 'greedy', 'kg', 'kg_indep',
       'okg', 'okg_indep', 'okg_ryzhov', 'randomized_greedy'].includes(p)
    : true;
  const effectivePolicy = policyAllowed(policy) ? policy : 'okg_ryzhov';

  const isHuman = effectivePolicy === 'human';
  const isKGFamily = ['kg', 'kg_indep', 'okg', 'okg_indep', 'okg_ryzhov'].includes(effectivePolicy);
  const isIE = effectivePolicy === 'ie';
  const isRandomizedGreedy = effectivePolicy === 'randomized_greedy';

  // Parsed numeric versions of the policy-parameter inputs.
  const mStar       = Math.max(1, Math.round(Number(mStarStr) || 1));
  const zAlpha      = Math.max(0, Number(zAlphaStr) || 0);
  const sigmaGreedy = Math.max(0, Number(sigmaGreedyStr) || 0);

  // θ search-box bounds — always sent as scalars for 1-D, as 2-tuples for 2-D.
  const impparamMin = is2D
    ? [numeric('theta1_min'), numeric('theta2_min')]
    : numeric('theta1_min');
  const impparamMax = is2D
    ? [numeric('theta1_max'), numeric('theta2_max')]
    : numeric('theta1_max');

  // Convert user-visible median-jump-in-dollars back into the
  // log-fraction the backend expects. AUM comes from the editable
  // Advanced-params row (falls back to the 1-D default). For 2-D
  // the AUM base is institutional (initial_aum × 0.5 by default).
  // Guards against ≤ 0 (Math.log = −Infinity / NaN) by falling back
  // to defaults.
  const _initial_aum = Math.max(1, numeric('initial_aum'));
  const _log_from_median = (key, aum) => {
    const raw = Number(adv[key]);
    const md  = (raw > 0) ? raw : Number(ADV_DEFAULTS[key]);
    return Math.log(md / aum);
  };

  // sim_config payload — field names are app-specific. Unknown fields are
  // dropped server-side; we just include everything the target app understands.
  // Institutional AUM for 2-D — uses the default institutional
  // fraction (0.5) applied to the editable total AUM. If the fraction
  // ever becomes user-editable, thread it through here too.
  const _aum_inst_2d = _initial_aum * 0.5;

  const simConfigPayload = is2D
    ? {
        stationary,
        initial_aum:           _initial_aum,
        impparam_min: impparamMin,
        impparam_max: impparamMax,
        // Objective / market
        r_market_annual:       numeric('r_market_annual'),
        r_cash_annual:         numeric('r_cash_annual'),
        trading_days_per_year: Math.round(numeric('trading_days_per_year')),
        // Individual investor process
        mu_ind_annual:         numeric('mu_ind_annual'),
        sigma_ind_annual:      numeric('sigma_ind_annual'),
        r_borrow_ind_annual:   numeric('r_borrow_ind_annual'),
        // Institutional investor process
        jump_rate_inst_annual: numeric('jump_rate_inst_annual'),
        jump_mean_log_inst:    _log_from_median('jump_median_dollars_inst', _aum_inst_2d),
        jump_std_log_inst:     numeric('jump_std_log_inst'),
        r_borrow_inst_annual:  numeric('r_borrow_inst_annual'),
      }
    : {
        stationary,
        initial_aum:      _initial_aum,
        r_market_annual:  numeric('r_market_annual'),
        r_cash_annual:    numeric('r_cash_annual'),
        jump_rate_annual: numeric('jump_rate_annual'),
        jump_mean_log:    _log_from_median('jump_median_dollars', _initial_aum),
        jump_std_log:     numeric('jump_std_log'),
        sigma_net_annual: numeric('sigma_net_annual'),
        r_borrow_annual:  numeric('r_borrow_annual'),
        impparam_min:     impparamMin,
        impparam_max:     impparamMax,
      };

  // For 2-D belief prior, broadcast the scalar length_scale into a per-dim
  // pair with the second dimension having a wider length scale (the
  // institutional-buffer axis varies over [0.01, 0.40] — 2× the individual
  // buffer's range — so a matching wider length scale is a sensible default).
  const beliefConfigPayload = is2D
    ? {
        length_scale: [numeric('length_scale'), 2 * numeric('length_scale')],
        signal_std:   numeric('signal_std'),
        noise_std:    numeric('noise_std'),
        prior_mean:   numeric('prior_mean'),
      }
    : {
        length_scale: numeric('length_scale'),
        signal_std:   numeric('signal_std'),
        noise_std:    numeric('noise_std'),
        prior_mean:   numeric('prior_mean'),
      };

  // Serialise the current form state into a panel blob.
  function currentFormBlob() {
    return { seed, stationary, mStar, zAlpha, sigmaGreedy, flowHorizon, reportLevel, adv };
  }

  // Write the current form state into the ACTIVE panel of the store,
  // persist to localStorage, and keep the legacy single-blob key in
  // sync (used as auto-launch fallback if the panels store is ever
  // wiped). Also updates React state so the selector reflects the
  // save immediately.
  function persistAdvanced(overrideActive) {
    const activeName = overrideActive ?? activePanelName;
    const blob = currentFormBlob();
    setPanelsStore(prev => {
      const nextForApp = {
        active: activeName,
        panels: { ...(prev?.[initialAppName]?.panels ?? {}), [activeName]: blob },
      };
      const nextStore = { ...prev, [initialAppName]: nextForApp };
      writePanelsStore(nextStore);
      return nextStore;
    });
    writeSavedAdvanced(blob);   // legacy fallback
  }

  // Load a named panel: populate every form field from its saved
  // values (falling back to ADV_DEFAULTS for anything missing so old
  // panels get new fields automatically) and mark it active.
  function loadPanel(name) {
    const blob = panelsStore?.[initialAppName]?.panels?.[name];
    if (!blob) return;
    setSeed(blob.seed ?? 42);
    setStationary(blob.stationary ?? true);
    setMStarStr(blob.mStar       != null ? String(blob.mStar)       : '1');
    setZAlphaStr(blob.zAlpha     != null ? String(blob.zAlpha)      : '0');
    setSigmaGreedyStr(blob.sigmaGreedy != null ? String(blob.sigmaGreedy) : '0');
    setFlowHorizonStr(blob.flowHorizon != null ? String(blob.flowHorizon) : '200');
    setReportLevel(blob.reportLevel ?? 'basic');
    setAdv({ ...ADV_DEFAULTS, ...(blob.adv ?? {}) });
    setPanelsStore(prev => {
      const nextStore = {
        ...prev,
        [initialAppName]: { ...prev[initialAppName], active: name },
      };
      writePanelsStore(nextStore);
      return nextStore;
    });
  }

  // Warm the backend reveal cache for the current form values.
  // Creates a temp session, calls /reveal (which is memoized on the
  // backend by a hash of sim_config + seed + horizon + grid + n_reps),
  // and deletes the temp session. Cost: a few seconds one-time. Payoff:
  // when the user next hits Play, the auto-bg-fetch on session-create
  // hits the warm cache, and the very first Reveal-truth click returns
  // instantly instead of blocking on a n_reps=50 sweep. Silent on
  // failure — the reveal is a nice-to-have, not required for the game
  // to work.
  async function precomputeReveal() {
    // 2-D reveal isn't wired up (we render a KG heatmap instead), so
    // don't waste cycles precomputing it.
    if (is2D) return;
    setPrecomputing(true);
    let sid = null;
    try {
      const acqConfigPayload =
        isIE               ? { z_alpha: zAlpha } :
        isRandomizedGreedy ? { sigma_greedy: sigmaGreedy } :
        {};
      const resp = await createSession({
        app_name: appName,
        policy: effectivePolicy,
        session_seed: seed,
        sim_config: simConfigPayload,
        belief_config: beliefConfigPayload,
        acq_config: acqConfigPayload,
        session_config: { horizon_weeks: horizon },
        budget: Math.max(1, Math.round(numeric('ryzhov_budget'))),
        m_star: isKGFamily ? mStar : 1,
        report_level: reportLevel,
        flow_horizon: flowHorizon,
        initial_theta: is2D
          ? [numeric('initial_theta1'), numeric('initial_theta2')]
          : numeric('initial_theta'),
      });
      sid = resp?.session_id;
      if (sid) await getReveal(sid);
    } catch (err) {
      console.warn('[precomputeReveal] skipped due to error', err);
    } finally {
      if (sid) { try { await deleteSession(sid); } catch (_) { /* ignore */ } }
      setPrecomputing(false);
    }
  }

  // "Save as new" — prompt for a name, save the current form state
  // as a new panel under that name (or overwrite if the user confirms).
  // Stays on the form (unlike Save-and-exit) so the user can keep
  // tweaking.
  async function handleSaveAsNew() {
    const raw = window.prompt('Panel name?');
    if (raw == null) return;
    const name = raw.trim();
    if (!name) return;
    const existing = panelsStore?.[initialAppName]?.panels ?? {};
    if (existing[name] && !window.confirm(`Panel "${name}" already exists. Overwrite?`)) {
      return;
    }
    persistAdvanced(name);
    await precomputeReveal();
  }

  // Rename the currently-active panel. Refuses to rename "Default"
  // (it's the always-present base) or to an existing name.
  function handleRenamePanel() {
    if (activePanelName === DEFAULT_PANEL_NAME) return;
    const raw = window.prompt(`Rename "${activePanelName}" to:`, activePanelName);
    if (raw == null) return;
    const newName = raw.trim();
    if (!newName || newName === activePanelName) return;
    const existing = panelsStore?.[initialAppName]?.panels ?? {};
    if (existing[newName]) {
      window.alert(`Panel "${newName}" already exists. Pick a different name.`);
      return;
    }
    setPanelsStore(prev => {
      const oldPanels = prev?.[initialAppName]?.panels ?? {};
      const nextPanels = {};
      // Preserve insertion order but swap the key of the active panel.
      for (const [k, v] of Object.entries(oldPanels)) {
        nextPanels[k === activePanelName ? newName : k] = v;
      }
      const nextStore = {
        ...prev,
        [initialAppName]: { active: newName, panels: nextPanels },
      };
      writePanelsStore(nextStore);
      return nextStore;
    });
  }

  // Delete the currently-active panel and fall back to Default.
  // Also refuses to delete Default.
  function handleDeletePanel() {
    if (activePanelName === DEFAULT_PANEL_NAME) return;
    if (!window.confirm(`Delete panel "${activePanelName}"? This cannot be undone.`)) return;
    setPanelsStore(prev => {
      const { [activePanelName]: _dropped, ...remaining } = prev?.[initialAppName]?.panels ?? {};
      const nextStore = {
        ...prev,
        [initialAppName]: { active: DEFAULT_PANEL_NAME, panels: remaining },
      };
      writePanelsStore(nextStore);
      return nextStore;
    });
    // Load Default so the form re-syncs to its saved values.
    const defaultBlob = panelsStore?.[initialAppName]?.panels?.[DEFAULT_PANEL_NAME] ?? {};
    setSeed(defaultBlob.seed ?? 42);
    setStationary(defaultBlob.stationary ?? true);
    setMStarStr(defaultBlob.mStar       != null ? String(defaultBlob.mStar)       : '1');
    setZAlphaStr(defaultBlob.zAlpha     != null ? String(defaultBlob.zAlpha)      : '0');
    setSigmaGreedyStr(defaultBlob.sigmaGreedy != null ? String(defaultBlob.sigmaGreedy) : '0');
    setFlowHorizonStr(defaultBlob.flowHorizon != null ? String(defaultBlob.flowHorizon) : '200');
    setReportLevel(defaultBlob.reportLevel ?? 'basic');
    setAdv({ ...ADV_DEFAULTS, ...(defaultBlob.adv ?? {}) });
  }

  // Auto-launch path: called by the useEffect when the URL has
  // ?auto=1 (the landing page's Play button). Persists first so the
  // in-game session and the on-disk record stay in sync.
  async function handleAutoLaunch() {
    setLoading(true);
    try {
      persistAdvanced();
      // acq_config carries the IE's z_alpha (θ^IE) or RandomizedGreedy's
      // sigma_greedy — server-side defaults are 0, so we only send the
      // one that applies to the selected policy.
      const acqConfigPayload =
        isIE               ? { z_alpha: zAlpha } :
        isRandomizedGreedy ? { sigma_greedy: sigmaGreedy } :
        {};
      await onCreate({
        app_name: appName,
        policy: effectivePolicy,
        session_seed: seed,
        sim_config: simConfigPayload,
        belief_config: beliefConfigPayload,
        acq_config: acqConfigPayload,
        session_config: { horizon_weeks: horizon },
        // Ryzhov N — used by OKGRyzhovCorrelatedPolicy; ignored by
        // every other policy. Backend defaults to 10 if we send null.
        budget: Math.max(1, Math.round(numeric('ryzhov_budget'))),
        // KG-family m* (θ^KGm*, in days). Ignored by non-KG policies.
        m_star: isKGFamily ? mStar : 1,
        report_level: reportLevel,
        flow_horizon: flowHorizon,
        // Starting θ for the ExperimentBar's Starting-point box and
        // the initial cash-balance sample-path fetch. 2-D uses a
        // 2-vector; 1-D a scalar.
        initial_theta: is2D
          ? [numeric('initial_theta1'), numeric('initial_theta2')]
          : numeric('initial_theta'),
      });
    } finally {
      setLoading(false);
    }
  }

  // Button-click path: this panel is an edit-and-exit report, not a
  // gateway into the game. Save the form values so a subsequent
  // Play-the-game click uses them, then return to the landing page.
  // Precomputes the true-reward curve on the way out so the first
  // in-game "Reveal truth" click is instant.
  async function handleSaveAndExit(e) {
    if (e && e.preventDefault) e.preventDefault();
    persistAdvanced();
    await precomputeReveal();

    // Three exit paths, in order of preference:
    //   1. If we came from the game itself (in-game "Game parameters"
    //      button set a sessionStorage flag), spawn a fresh game with
    //      the new params. Streamlines Warren's "tweak → replay" loop.
    //   2. If the referrer is the landing page, history.back() —
    //      browser natively restores scroll position.
    //   3. Otherwise fall through to a fresh landing navigation.
    let fromGame = false;
    try {
      fromGame = sessionStorage.getItem('lwd_from_game') === '1';
      if (fromGame) sessionStorage.removeItem('lwd_from_game');
    } catch (_) { /* private mode — treat as false */ }
    if (fromGame) {
      // Fresh game session with the new params (auto=1 spawns it).
      const gameUrl = `${window.location.origin}/?app=${encodeURIComponent(appName)}&auto=1`;
      window.location.href = gameUrl;
      return;
    }

    // Referrer detection: browsers default to "strict-origin-when-
    // cross-origin", so cross-origin referrer is only the origin
    // (https://warrenpowell.org/), not the full landing URL. Match
    // by hostname, not by the full path.
    let cameFromLanding = false;
    try {
      cameFromLanding = document.referrer &&
        new URL(document.referrer).hostname === 'warrenpowell.org';
    } catch (_) { /* malformed referrer — treat as "no" */ }

    // Build the landing URL — carries the app choice, the list of
    // saved panel names for that app, and the currently-active panel
    // name, so the landing page's picker can mirror our state. Only
    // used on the fresh-navigation branch; history.back doesn't
    // refresh landing (it re-uses landing's own localStorage instead).
    let url;
    try {
      url = new URL(LANDING_URL);
      url.searchParams.set('app', appName);
      const names = Object.keys(panelsStore?.[initialAppName]?.panels ?? {});
      if (names.length > 0) url.searchParams.set('panels', names.join(','));
      if (activePanelName) url.searchParams.set('active', activePanelName);
      url = url.toString();
    } catch { url = LANDING_URL; }
    if (cameFromLanding && window.history.length > 1) {
      window.history.back();
    } else {
      window.location.href = url;
    }
  }

  // Auto-submit on mount when the landing page's Play button was hit
  // (URL had ?auto=1). Only fires once — subsequent renders of this
  // form (e.g. after a New session click) will not re-fire because
  // React reuses the same instance. Uses the stable defaults for
  // everything except appName/policy carried in as props.
  const autoFiredRef = useRef(false);
  useEffect(() => {
    if (!autoSubmit || autoFiredRef.current) return;
    autoFiredRef.current = true;
    handleAutoLaunch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSubmit]);

  // Reusable renderer for a single advanced-parameter input
  const advField = ([key, label, hint]) => (
    <div className="form-group" key={key}>
      <label style={{ fontSize: 12 }}>{label}</label>
      <input
        type="number"
        value={adv[key]}
        step="any"
        onChange={e => setField(key, e.target.value)}
        onBlur={() => canonicalize(key)}
      />
      <span style={{ fontSize: 11, color: '#94a3b8' }}>{hint}</span>
    </div>
  );

  // Human-readable summary of the choice carried through from the
  // landing page — so the user knows what they're about to Start.
  const appLabel    = APPS.find(a => a.value === appName)?.label ?? appName;
  const policyLabel = ({
    human: 'Manual — I pick θ each round',
    randomized_greedy: 'Randomized greedy',
    kg: 'Offline KG',
    kg_indep: 'KG — offline independent',
    okg: 'KG — online correlated (μ + KG(ρˡᵏʰᵈ))',
    okg_indep: 'KG — online independent',
    okg_ryzhov: 'KG — online correlated — Ryzhov (μ + (N−n)·KG(ρˡᵏʰᵈ))',
    ie: 'IE',
    ie_15: 'IE (ρ^IE = 1.5)',
    greedy: 'Greedy',
    random: 'Random — baseline',
  })[effectivePolicy] ?? effectivePolicy;

  // When auto-launching from the landing page, hide the form entirely
  // and show a brief starting-up notice — the user shouldn't see the
  // Advanced-parameters panel unless they asked for it.
  if (autoSubmit) {
    return (
      <div className="setup-wrap">
        <div className="card setup-card" style={{ textAlign: 'center', padding: 32 }}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#0f172a', marginBottom: 8 }}>
            Starting session…
          </div>
          <div style={{ fontSize: 13, color: '#64748b' }}>
            {appLabel} · {policyLabel}
          </div>
          {error && (
            <div style={{ marginTop: 16, color: '#b91c1c', fontSize: 13 }}>
              {String(error?.message ?? error)}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Row schema for the current app.
  const rows = is2D ? ROWS_2D : ROWS_1D;

  // Getters / setters for a source path (`reportLevel`, `seed`, `mStarStr`,
  // or `adv:<key>`) — keeps the row renderer decoupled from state layout.
  const getVal = (src) => {
    if (src.startsWith('adv:')) return adv[src.slice(4)] ?? '';
    if (src === 'reportLevel') return reportLevel;
    if (src === 'seed')        return String(seed);
    if (src === 'mStarStr')    return mStarStr;
    return '';
  };
  const setVal = (src, v) => {
    if (src.startsWith('adv:')) return setField(src.slice(4), v);
    if (src === 'reportLevel') return setReportLevel(v);
    if (src === 'seed')        return setSeed(Number(v) || 0);
    if (src === 'mStarStr')    return setMStarStr(v);
  };
  // Canonicalize on blur: for adv fields, snap to numeric; for seed, coerce int.
  const blurVal = (src) => {
    if (src.startsWith('adv:')) return canonicalize(src.slice(4));
    if (src === 'seed')        return setSeed(Math.max(0, Math.min(9999, Math.round(Number(seed) || 0))));
    if (src === 'mStarStr')    return setMStarStr(String(Math.max(1, Math.min(100, Math.round(Number(mStarStr) || 1)))));
  };

  return (
    <div className="setup-wrap" style={{ alignItems: 'flex-start', paddingTop: 24 }}>
      <div className="card" style={{ maxWidth: 1100, width: '100%' }}>
        <h1 className="form-title">Game parameters</h1>
        <p className="form-subtitle">
          Edit any value in the <b>Default</b> column. Save and exit
          returns you to the landing page — the parameter-adjustment
          policy is chosen inside the game on the control bar.
        </p>

        <form onSubmit={handleSaveAndExit}>
          {/* Admin bar — everything a user might want to click sits
              here at the top: app label, panel picker + Rename/Delete,
              and the primary Save actions. Puts all controls in one
              place so a first-time visitor sees them immediately
              instead of scrolling to the bottom of a long table. */}
          <div style={{ padding: '12px 14px', background: '#f8fafc',
                        border: '1px solid #e2e8f0', borderRadius: 6,
                        marginBottom: 8, fontSize: 13, color: '#374151',
                        display: 'flex', gap: 16, alignItems: 'center',
                        flexWrap: 'wrap', justifyContent: 'space-between' }}>
            {/* Left group — what you're editing */}
            <div style={{ display: 'flex', gap: 18, alignItems: 'center', flexWrap: 'wrap' }}>
              <span>
                <span style={{ color: '#64748b' }}>Cash management policy:</span> <b>{appLabel}</b>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                <span style={{ color: '#64748b' }}>Panel:</span>
                <select value={activePanelName}
                        onChange={e => loadPanel(e.target.value)}
                        style={{ padding: '3px 6px', border: '1px solid #cbd5e1',
                                 borderRadius: 4, fontSize: 13, background: '#fff' }}>
                  {panelNameList(panelsStore, initialAppName).map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                <button type="button" onClick={handleRenamePanel}
                        disabled={activePanelName === DEFAULT_PANEL_NAME}
                        title={activePanelName === DEFAULT_PANEL_NAME
                          ? 'The Default panel cannot be renamed'
                          : `Rename "${activePanelName}"`}
                        style={{ padding: '3px 10px', border: '1px solid #cbd5e1',
                                 borderRadius: 4, fontSize: 12, background: '#fff',
                                 cursor: activePanelName === DEFAULT_PANEL_NAME ? 'not-allowed' : 'pointer',
                                 opacity: activePanelName === DEFAULT_PANEL_NAME ? 0.5 : 1 }}>
                  Rename
                </button>
                <button type="button" onClick={handleDeletePanel}
                        disabled={activePanelName === DEFAULT_PANEL_NAME}
                        title={activePanelName === DEFAULT_PANEL_NAME
                          ? 'The Default panel cannot be deleted'
                          : `Delete "${activePanelName}"`}
                        style={{ padding: '3px 10px', border: '1px solid #cbd5e1',
                                 borderRadius: 4, fontSize: 12, background: '#fff',
                                 color: '#b91c1c',
                                 cursor: activePanelName === DEFAULT_PANEL_NAME ? 'not-allowed' : 'pointer',
                                 opacity: activePanelName === DEFAULT_PANEL_NAME ? 0.5 : 1 }}>
                  Delete
                </button>
              </span>
            </div>
            {/* Right group — save actions */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <button type="submit" className="btn btn-primary"
                      disabled={loading || precomputing}
                      style={{ padding: '7px 18px', fontSize: 13 }}
                      title={`Save changes into the current panel "${activePanelName}" and exit.`}>
                {precomputing ? 'Computing true curve…' : 'Save and exit'}
              </button>
              <button type="button" className="btn btn-outline"
                      onClick={handleSaveAsNew}
                      disabled={loading || precomputing}
                      style={{ padding: '7px 14px', fontSize: 13 }}
                      title="Save the current form values under a new panel name (leaves you on this page).">
                {precomputing ? 'Computing…' : 'Save as new…'}
              </button>
            </div>
          </div>
          {/* One-line legend under the admin bar so newcomers can
              tell the four buttons apart without hovering. */}
          <p style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 16px 2px' }}>
            <b>Save and exit</b> writes to "{activePanelName}" and returns
            to the landing page (precomputes the true-reward curve so the
            in-game <b>Reveal truth</b> click is instant — adds a few seconds).
            &nbsp;·&nbsp; <b>Save as new</b> creates a separate named panel.
            &nbsp;·&nbsp; <b>Rename</b>/<b>Delete</b> act on the panel currently
            in the dropdown (Default is protected).
          </p>

          <table style={{
            width: '100%', borderCollapse: 'collapse',
            fontSize: 13, tableLayout: 'fixed',
          }}>
            <colgroup>
              <col style={{ width: '32%' }} />
              <col style={{ width: '14%' }} />
              <col style={{ width: '14%' }} />
              <col style={{ width: '40%' }} />
            </colgroup>
            <thead>
              <tr style={{ borderBottom: '2px solid #cbd5e1' }}>
                <th style={TH}>Variable</th>
                <th style={TH}>Default</th>
                <th style={TH}>Range</th>
                <th style={TH}>Explanation</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) =>
                row.kind === 'section' ? (
                  <SectionRow key={`s-${i}`} label={row.label} />
                ) : (
                  <ParamRow key={row.source ?? (row.sources || []).join('-')}
                            row={row} getVal={getVal} setVal={setVal}
                            blurVal={blurVal} />
                )
              )}
            </tbody>
          </table>

          {error && (
            <p style={{ color: '#dc2626', fontSize: 13, margin: '16px 0 0 0' }}>{error}</p>
          )}
        </form>
      </div>
    </div>
  );
}

// ── Table sub-components ────────────────────────────────────────────────

const TH = {
  padding: '8px 10px', textAlign: 'left', fontSize: 12, fontWeight: 700,
  color: '#334155', letterSpacing: 0.3, background: '#f8fafc',
};
const TD = { padding: '8px 10px', verticalAlign: 'top', borderBottom: '1px solid #e2e8f0' };
const TD_EDIT = { ...TD, background: '#fefce8' };  // highlight editable cells
const TD_DESC = { ...TD, color: '#64748b', fontSize: 12, lineHeight: 1.5 };

function SectionRow({ label }) {
  return (
    <tr>
      <td colSpan={4} style={{
        padding: '10px 10px 6px 10px',
        fontSize: 12, fontWeight: 700, color: '#0f172a',
        textTransform: 'uppercase', letterSpacing: 0.6,
        background: '#f1f5f9', borderBottom: '1px solid #cbd5e1',
      }}>
        {label}
      </td>
    </tr>
  );
}

function ParamRow({ row, getVal, setVal, blurVal }) {
  return (
    <tr>
      <td style={{ ...TD, fontWeight: 500, color: '#0f172a' }}>{row.label}</td>
      <td style={TD_EDIT}>
        {row.kind === 'select' ? (
          <select value={getVal(row.source)}
                  onChange={e => setVal(row.source, e.target.value)}
                  style={SELECT_STYLE}>
            {row.options.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        ) : row.kind === 'pair' ? (
          <div style={{ display: 'flex', gap: 6 }}>
            {row.sources.map((s, i) => (
              <input key={i} type="number"
                     value={getVal(s)}
                     min={row.min} step={row.step ?? 'any'}
                     onChange={e => setVal(s, e.target.value)}
                     onBlur={() => blurVal(s)}
                     style={INPUT_STYLE_SMALL} />
            ))}
          </div>
        ) : (
          <input type="number"
                 value={getVal(row.source)}
                 min={row.min} max={row.max}
                 step={row.kind === 'int' ? 1 : (row.step ?? 'any')}
                 onChange={e => setVal(row.source, e.target.value)}
                 onBlur={() => blurVal(row.source)}
                 style={INPUT_STYLE} />
        )}
      </td>
      <td style={{ ...TD, color: '#64748b', fontSize: 12, fontFamily: 'ui-monospace, monospace' }}>
        {row.range}
      </td>
      <td style={TD_DESC}>{row.desc}</td>
    </tr>
  );
}

const INPUT_STYLE = {
  width: '100%', boxSizing: 'border-box',
  padding: '4px 8px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 13, background: '#fff',
};
const INPUT_STYLE_SMALL = { ...INPUT_STYLE, width: '48%' };
const SELECT_STYLE = {
  width: '100%', boxSizing: 'border-box',
  padding: '4px 8px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 13, background: '#fff',
};
