import { useState } from 'react';

// Defaults must match backend Pydantic defaults (shell/models.py) so the
// initial UI state is consistent with what the API would use for an empty
// request. All values are held as strings during typing and coerced to
// numbers on submit; onBlur canonicalizes ("050" → "50").
const ADV_DEFAULTS = {
  // Belief prior — the GP's assumptions before it sees any data
  length_scale:      '0.04',
  signal_std:        '5000',
  noise_std:         '3000',
  prior_mean:        '5000',
  // Simulation model — the underlying truth F(θ) and its per-run noise
  jump_rate_annual:  '12',
  jump_std_log:      '0.5',
  sigma_net_annual:  '0.02',
  r_borrow_annual:   '0.10',
};

// Field metadata for the advanced-parameters panel
const BELIEF_FIELDS = [
  ['length_scale', 'Length scale ℓ',
    'Smoothness of prior F(θ). Larger = fewer wiggles.'],
  ['signal_std', 'Signal std σ_f',
    'Prior amplitude of F(θ) in cost units.'],
  ['noise_std', 'Noise std σ_n',
    'Per-observation noise the GP assumes (algorithm side).'],
  ['prior_mean', 'Prior mean m₀',
    'Constant prior mean of F everywhere.'],
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
  const policyAllowed = (p) => is2D ? ['random', 'ie', 'kg'].includes(p) : true;
  const effectivePolicy = policyAllowed(policy) ? policy : 'kg';

  const isHuman = effectivePolicy === 'human';
  const isBatch = effectivePolicy === 'kg-batch' || effectivePolicy === 'ie-batch';
  const family  = effectivePolicy === 'kg-batch' ? 'KG' : effectivePolicy === 'ie-batch' ? 'IE' : null;

  // For 2-D apps, don't send single-value-tuple sim_config knobs that would
  // clash with the app's 2-parameter defaults.  We ship only stationary; the
  // 2-D app's own jump / flow / borrow-rate defaults take over.
  const simConfigPayload = is2D
    ? { stationary }
    : {
        stationary,
        jump_rate_annual: numeric('jump_rate_annual'),
        jump_std_log:     numeric('jump_std_log'),
        sigma_net_annual: numeric('sigma_net_annual'),
        r_borrow_annual:  numeric('r_borrow_annual'),
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
        budget: (isHuman || isBatch) ? budget : null,
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

          {(isHuman || isBatch) && (
            <div className="form-group">
              <label>{isBatch ? 'Budget (steps per policy per sim)' : 'Adjustment budget'}</label>
              <input
                type="number" value={budgetStr} min={1} max={50}
                onChange={e => setBudgetStr(e.target.value)}
                onBlur={() => setBudgetStr(String(budget))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                {isBatch
                  ? 'How many observations each policy can take before we score it'
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

              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6,
                            marginTop: 16, marginBottom: 8 }}>
                Simulation model (underlying truth &amp; per-run noise)
              </div>
              <div className="form-row">{SIM_FIELDS.slice(0, 2).map(advField)}</div>
              <div className="form-row">{SIM_FIELDS.slice(2, 4).map(advField)}</div>

              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 12, marginBottom: 0 }}>
                Simulation parameters change what F(θ) actually looks like and how noisy
                a single 26-week run is. Belief parameters are what the GP algorithms
                <em> assume</em> — mismatch between the two is itself a pedagogically
                interesting condition.
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
