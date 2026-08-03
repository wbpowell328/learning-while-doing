import { useState } from 'react';

// Defaults must match backend Pydantic defaults (shell/models.py) so the
// initial UI state is consistent with what the API would use for an empty
// request. All values are held as strings during typing and coerced to
// numbers on submit; onBlur canonicalizes ("050" → "50").
const ADV_DEFAULTS = {
  // Belief prior — the GP's assumptions before it sees any data.
  // Values are in the *reward* frame (typical run reward ~$35k, varies
  // ~$1-3k across θ, per-run noise ~$1-2k). Session negates prior_mean
  // internally so the belief layer keeps operating on "values to
  // minimise" (=-reward); signal_std and noise_std are variances so are
  // frame-independent.
  length_scale:      '0.04',
  signal_std:        '3000',
  noise_std:         '1500',
  prior_mean:        '35000',
  // Simulation model (1-D app only) — the underlying truth F(θ) and its per-run noise
  jump_rate_annual:  '12',
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
  jump_mean_log_inst:    '-2.5',   // median jump ~exp(-2.5) ≈ 8% of aum_inst
  jump_std_log_inst:     '0.6',    // log-space std of jump sizes
  r_borrow_inst_annual:  '0.02',   // 2% redemption fee on forced institutional liquidation
};

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

export default function SessionForm({ onCreate, error }) {
  const [appName, setAppName]       = useState('cash_balance');
  const [policy, setPolicy]         = useState('human');
  const [seed, setSeed]             = useState(42);
  const [horizon, setHorizon]       = useState(26);
  const [stationary, setStationary] = useState(true);
  const [loading, setLoading]       = useState(false);

  const [budgetStr, setBudgetStr] = useState('10');
  const [simsStr,   setSimsStr]   = useState('10');
  const budget = Math.max(1, Math.min(50, Number(budgetStr) || 10));
  const simsPerPolicy = Math.max(1, Math.min(100, Number(simsStr) || 10));

  // Advanced parameters — kept as strings during typing.
  const [adv, setAdv] = useState(ADV_DEFAULTS);
  const setField     = (k, v) => setAdv(prev => ({ ...prev, [k]: v }));
  const canonicalize = (k) => setField(k, String(Number(adv[k]) || Number(ADV_DEFAULTS[k])));
  const numeric = (k) => Number(adv[k]) || Number(ADV_DEFAULTS[k]);

  // Selected app metadata.
  const appMeta = APPS.find(a => a.value === appName) ?? APPS[0];
  const is2D = appMeta.dim >= 2;

  // Auto-flip incompatible policy selections when app changes.
  // Human mode and batch modes are 1-D-only for now.
  const policyAllowed = (p) => is2D
    ? ['random', 'ie', 'kg', 'kg_indep', 'okg', 'okg_indep'].includes(p)
    : true;
  const effectivePolicy = policyAllowed(policy) ? policy : 'kg';

  const isHuman = effectivePolicy === 'human';
  const isBatch = effectivePolicy === 'kg-batch' || effectivePolicy === 'ie-batch';
  // Online-KG variants need N to compute (N−n) in the Ryzhov formula.
  const isOKG   = effectivePolicy === 'okg' || effectivePolicy === 'okg_indep';
  const family  = effectivePolicy === 'kg-batch' ? 'KG' : effectivePolicy === 'ie-batch' ? 'IE' : null;

  // θ search-box bounds — always sent as scalars for 1-D, as 2-tuples for 2-D.
  const impparamMin = is2D
    ? [numeric('theta1_min'), numeric('theta2_min')]
    : numeric('theta1_min');
  const impparamMax = is2D
    ? [numeric('theta1_max'), numeric('theta2_max')]
    : numeric('theta1_max');

  // sim_config payload — field names are app-specific. Unknown fields are
  // dropped server-side; we just include everything the target app understands.
  const simConfigPayload = is2D
    ? {
        stationary,
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
        jump_mean_log_inst:    numeric('jump_mean_log_inst'),
        jump_std_log_inst:     numeric('jump_std_log_inst'),
        r_borrow_inst_annual:  numeric('r_borrow_inst_annual'),
      }
    : {
        stationary,
        jump_rate_annual: numeric('jump_rate_annual'),
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

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    try {
      await onCreate({
        app_name: appName,
        policy: effectivePolicy,
        session_seed: seed,
        sim_config: simConfigPayload,
        belief_config: beliefConfigPayload,
        session_config: { horizon_weeks: horizon },
        budget: (isHuman || isBatch || isOKG) ? budget : null,
        family,
        sims_per_policy: simsPerPolicy,
      });
    } finally {
      setLoading(false);
    }
  }

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

  return (
    <div className="setup-wrap">
      <div className="card setup-card">
        <h1 className="form-title">Learning While Doing</h1>
        <p className="form-subtitle">
          Find the optimal cash buffer ratio θ for a mutual fund.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Application</label>
            <select value={appName} onChange={e => setAppName(e.target.value)}>
              {APPS.map(a => (
                <option key={a.value} value={a.value}>{a.label}</option>
              ))}
            </select>
            {is2D && (
              <span style={{ fontSize: 12, color: '#64748b' }}>
                2-parameter app — Human mode and batch benchmarks are 1-D only for now;
                automated modes (KG, IE, Random) work here.
              </span>
            )}
          </div>

          <div className="form-group">
            <label>Mode</label>
            <select value={effectivePolicy} onChange={e => setPolicy(e.target.value)}>
              <optgroup label="Single run">
                {!is2D && <option value="human">Human — I pick θ each round</option>}
                <option value="kg">KG — offline correlated (analytic)</option>
                <option value="kg_indep">KG — offline independent</option>
                <option value="okg">KG — online correlated (Ryzhov)</option>
                <option value="okg_indep">KG — online independent</option>
                <option value="ie">IE — LCB with z_alpha=0 (greedy)</option>
                <option value="random">Random — baseline</option>
              </optgroup>
              {!is2D && (
                <optgroup label="Batch benchmark">
                  <option value="kg-batch">KG batch — all 5 variants</option>
                  <option value="ie-batch">IE batch — 21 z_alpha values</option>
                </optgroup>
              )}
            </select>
          </div>

          {(isHuman || isBatch || isOKG) && (
            <div className="form-group">
              <label>{isBatch ? 'Budget (steps per policy per sim)'
                     : isOKG   ? 'Measurement budget N'
                     :           'Adjustment budget'}</label>
              <input
                type="number" value={budgetStr} min={1} max={50}
                onChange={e => setBudgetStr(e.target.value)}
                onBlur={() => setBudgetStr(String(budget))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                {isBatch
                  ? 'How many observations each policy can take before we score it'
                  : isOKG
                  ? 'Online KG uses N to compute the info-value bonus (N−n)·KG(x)'
                  : 'Number of times you can run the simulator'}
              </span>
            </div>
          )}

          {isBatch && (
            <div className="form-group">
              <label>Simulations per policy</label>
              <input
                type="number" value={simsStr} min={1} max={100}
                onChange={e => setSimsStr(e.target.value)}
                onBlur={() => setSimsStr(String(simsPerPolicy))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                {family === 'KG'
                  ? `Total runs = 5 policies × ${simsPerPolicy} sims × ${budget} steps = ${5 * simsPerPolicy * budget} simulations`
                  : `Total runs = 21 policies × ${simsPerPolicy} sims × ${budget} steps = ${21 * simsPerPolicy * budget} simulations`}
              </span>
            </div>
          )}

          <div className="form-row">
            <div className="form-group">
              <label>Random seed</label>
              <input type="number" value={seed} min={0} max={9999}
                onChange={e => setSeed(Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Weeks per run</label>
              <input type="number" value={horizon} min={1} max={52}
                onChange={e => setHorizon(Number(e.target.value))} />
            </div>
          </div>

          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <input type="checkbox" id="stationary" checked={stationary}
              onChange={e => setStationary(e.target.checked)}
              style={{ width: 16, height: 16, cursor: 'pointer' }} />
            <label htmlFor="stationary" style={{ cursor: 'pointer' }}>
              Stationary regime (no market regime switching)
            </label>
          </div>

          {/* Advanced parameters (collapsible) */}
          <details style={{ marginBottom: 16, borderTop: '1px solid #e2e8f0', paddingTop: 12 }}>
            <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: '#374151',
                              padding: '4px 0', userSelect: 'none' }}>
              Advanced parameters — belief prior & simulation truth
            </summary>
            <div style={{ marginTop: 12, padding: '12px 16px', background: '#f8fafc',
                          borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 8 }}>
                Belief prior (GP hyperparameters — algorithm assumptions)
              </div>
              <div className="form-row">{BELIEF_FIELDS.slice(0, 2).map(advField)}</div>
              <div className="form-row">{BELIEF_FIELDS.slice(2, 4).map(advField)}</div>

              {!is2D && (
                <>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                                textTransform: 'uppercase', letterSpacing: 0.6,
                                marginTop: 16, marginBottom: 8 }}>
                    Simulation model (underlying truth &amp; per-run noise)
                  </div>
                  <div className="form-row">{SIM_FIELDS.slice(0, 2).map(advField)}</div>
                  <div className="form-row">{SIM_FIELDS.slice(2, 4).map(advField)}</div>
                </>
              )}

              {is2D && (
                <>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                                textTransform: 'uppercase', letterSpacing: 0.6,
                                marginTop: 16, marginBottom: 8 }}>
                    Market rates &amp; calendar
                  </div>
                  <div className="form-row">{MARKET_FIELDS_2D.slice(0, 2).map(advField)}</div>
                  <div className="form-row">{MARKET_FIELDS_2D.slice(2, 3).map(advField)}</div>

                  <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                                textTransform: 'uppercase', letterSpacing: 0.6,
                                marginTop: 16, marginBottom: 8 }}>
                    Individual investors — GBM on aum_ind (small, frequent)
                  </div>
                  <div className="form-row">{IND_FIELDS_2D.slice(0, 2).map(advField)}</div>
                  <div className="form-row">{IND_FIELDS_2D.slice(2, 3).map(advField)}</div>

                  <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                                textTransform: 'uppercase', letterSpacing: 0.6,
                                marginTop: 16, marginBottom: 8 }}>
                    Institutional investors — Poisson × lognormal on aum_inst (rare, large)
                  </div>
                  <div className="form-row">{INST_FIELDS_2D.slice(0, 2).map(advField)}</div>
                  <div className="form-row">{INST_FIELDS_2D.slice(2, 4).map(advField)}</div>
                </>
              )}

              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6,
                            marginTop: 16, marginBottom: 8 }}>
                θ search-box bounds
              </div>
              {is2D ? (
                <>
                  <div className="form-row">{RANGE_FIELDS_2D.slice(0, 2).map(advField)}</div>
                  <div className="form-row">{RANGE_FIELDS_2D.slice(2, 4).map(advField)}</div>
                </>
              ) : (
                <div className="form-row">{RANGE_FIELDS_1D.map(advField)}</div>
              )}

              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 12, marginBottom: 0 }}>
                Simulation parameters change what F(θ) actually looks like and how noisy
                a single 26-week run is. Belief parameters are what the GP algorithms
                <em> assume</em> — mismatch between the two is itself a pedagogically
                interesting condition. Widening the θ box explores a larger region;
                make sure the length scale ℓ isn't tiny compared to the new range.
              </p>
            </div>
          </details>

          {error && (
            <p style={{ color: '#dc2626', fontSize: 13, marginBottom: 12 }}>{error}</p>
          )}

          <button type="submit" className="btn btn-primary" disabled={loading}
            style={{ width: '100%', padding: '11px', fontSize: '0.95rem' }}>
            {loading ? (isBatch ? 'Running batch…' : 'Creating…') : (isBatch ? 'Run batch →' : 'Start →')}
          </button>
        </form>
      </div>
    </div>
  );
}
