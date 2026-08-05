import { useState, useEffect, useRef } from 'react';

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

// Advanced-parameters are persisted in localStorage keyed by this string
// so the "Save and exit" flow is meaningful — the user's edits survive
// leaving the page and re-launching the game from the landing page.
// Bump the suffix if the shape of the saved blob ever changes.
const ADVANCED_STORAGE_KEY = 'lwd_advanced_v1';
// Where "Save and exit" sends the user back to. Landing page on the
// CASTLE site — the game is embedded there behind the Play button.
const LANDING_URL = 'https://warrenpowell.org/learning-while-doing/';

function readSavedAdvanced() {
  try {
    const raw = window.localStorage.getItem(ADVANCED_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch { return {}; }
}
function writeSavedAdvanced(blob) {
  try {
    window.localStorage.setItem(ADVANCED_STORAGE_KEY, JSON.stringify(blob));
  } catch { /* private mode / quota — silently accept the loss */ }
}

export default function SessionForm({
  onCreate, error,
  initialAppName = 'cash_balance',
  initialPolicy  = 'kg',
  autoSubmit     = false,   // when true (URL ?auto=1), fire submit on mount
}) {
  // App + policy come from the landing page via URL query params. The
  // dropdowns for them were moved to the landing page per Warren's spec,
  // so they aren't rendered here — but we still track the values so the
  // submit payload carries them through.
  const [appName]  = useState(initialAppName);
  const [policy]   = useState(initialPolicy);
  const [loading, setLoading] = useState(false);

  // Load any previously-saved advanced values so this panel is a
  // proper edit-and-exit surface: users get back exactly what they
  // last set. Anything not present in the blob falls through to the
  // built-in defaults below.
  const savedAdv = readSavedAdvanced();

  // Random seed + stationary regime — moved into the Advanced Parameters
  // panel per Warren. Weeks per run is dropped entirely (ExperimentBar's
  // "Run N days" input now sets the horizon on every iteration); we
  // still send a default horizon_weeks on session-create so the backend
  // SessionConfig has a value, but the user never sees it.
  const [seed, setSeed]             = useState(savedAdv.seed ?? 42);
  const [stationary, setStationary] = useState(savedAdv.stationary ?? true);
  const horizon = 26;   // backend default; overridden per-iteration by ExperimentBar

  // Policy parameter (single field, meaning depends on the policy):
  //   KG variants → m* (days) — how many repeat experiments the policy
  //     assumes when computing KG (batch-size trick).
  //   IE → z_alpha (# std devs) — LCB coefficient.
  //   Random / Human → unused.
  const [mStarStr,  setMStarStr]  = useState(savedAdv.mStar  != null ? String(savedAdv.mStar)  : '1');
  const [zAlphaStr, setZAlphaStr] = useState(savedAdv.zAlpha != null ? String(savedAdv.zAlpha) : '0');

  // Reporting level — controls how much of the diagnostic UI is shown.
  // Basic = core charts only; Advanced also shows the KG(x;m) card.
  const [reportLevel, setReportLevel] = useState(savedAdv.reportLevel ?? 'basic');

  // Advanced parameters — kept as strings during typing.
  const [adv, setAdv] = useState({ ...ADV_DEFAULTS, ...(savedAdv.adv ?? {}) });
  const setField     = (k, v) => setAdv(prev => ({ ...prev, [k]: v }));
  const canonicalize = (k) => setField(k, String(Number(adv[k]) || Number(ADV_DEFAULTS[k])));
  const numeric = (k) => Number(adv[k]) || Number(ADV_DEFAULTS[k]);

  // Selected app metadata.
  const appMeta = APPS.find(a => a.value === appName) ?? APPS[0];
  const is2D = appMeta.dim >= 2;

  // Auto-flip incompatible policy selections when app changes.
  const policyAllowed = (p) => is2D
    ? ['random', 'ie', 'kg', 'kg_indep', 'okg', 'okg_indep'].includes(p)
    : true;
  const effectivePolicy = policyAllowed(policy) ? policy : 'kg';

  const isHuman = effectivePolicy === 'human';
  const isKGFamily = ['kg', 'kg_indep', 'okg', 'okg_indep'].includes(effectivePolicy);
  const isIE = effectivePolicy === 'ie';

  // Parsed numeric versions of the policy-parameter inputs.
  const mStar  = Math.max(1, Math.round(Number(mStarStr) || 1));
  const zAlpha = Math.max(0, Number(zAlphaStr) || 0);

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

  // Serialise the current form state into the localStorage blob so
  // "Save and exit" (and every session-create) round-trips edits.
  function persistAdvanced() {
    writeSavedAdvanced({
      seed, stationary, mStar, zAlpha, reportLevel, adv,
    });
  }

  // Auto-launch path: called by the useEffect when the URL has
  // ?auto=1 (the landing page's Play button). Persists first so the
  // in-game session and the on-disk record stay in sync.
  async function handleAutoLaunch() {
    setLoading(true);
    try {
      persistAdvanced();
      // acq_config carries the IE's z_alpha (θ^IE). Server-side default
      // is 0 (greedy); we only send a nonzero value when IE is selected.
      const acqConfigPayload = isIE ? { z_alpha: zAlpha } : {};
      await onCreate({
        app_name: appName,
        policy: effectivePolicy,
        session_seed: seed,
        sim_config: simConfigPayload,
        belief_config: beliefConfigPayload,
        acq_config: acqConfigPayload,
        session_config: { horizon_weeks: horizon },
        // Adjustment-budget field was dropped from the UI (the
        // ExperimentBar's Restart / One more controls make it moot);
        // send null so the backend uses its own default if it ever
        // needs one.
        budget: null,
        // KG-family m* (θ^KGm*, in days). Ignored by non-KG policies.
        m_star: isKGFamily ? mStar : 1,
        report_level: reportLevel,
      });
    } finally {
      setLoading(false);
    }
  }

  // Button-click path: this panel is an edit-and-exit report, not a
  // gateway into the game. Save the form values so a subsequent
  // Play-the-game click uses them, then return to the landing page.
  function handleSaveAndExit(e) {
    if (e && e.preventDefault) e.preventDefault();
    persistAdvanced();
    window.location.href = LANDING_URL;
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
    human: 'Human — I pick θ each round',
    kg: 'KG — offline correlated (analytic)',
    kg_indep: 'KG — offline independent',
    okg: 'KG — online correlated',
    okg_indep: 'KG — online independent',
    ie: 'IE — LCB (upper-confidence exploration)',
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

  return (
    <div className="setup-wrap">
      <div className="card setup-card">
        <h1 className="form-title">Advanced parameters</h1>
        <p className="form-subtitle">
          Tune the belief prior, simulation truth, seed, policy
          parameters, and reporting level. Save and exit to return to
          the landing page — the parameter-adjustment policy is chosen
          inside the game on the control bar.
        </p>

        <form onSubmit={handleSaveAndExit}>
          {/* Show which cash-management app these edits are for. */}
          <div style={{ padding: '10px 12px', background: '#f8fafc',
                        border: '1px solid #e2e8f0', borderRadius: 6,
                        marginBottom: 16, fontSize: 13, color: '#374151' }}>
            <span style={{ color: '#64748b' }}>Cash management policy:</span> <b>{appLabel}</b>
          </div>

          {/* Policy parameter — the one non-hidden thing that varies
              by policy. Kept out here so it's visible without expanding
              Advanced. KG family: m*. IE: z_alpha. Random / Human: none. */}
          {isKGFamily && (
            <div className="form-group">
              <label>Policy parameter — m* (days)</label>
              <input
                type="number" value={mStarStr} min={1} step={1}
                onChange={e => setMStarStr(e.target.value)}
                onBlur={() => setMStarStr(String(mStar))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                KG evaluated as if we ran m* repeat experiments (precision ×m*).
                m*=1 recovers the classical single-shot KG. Tunable mid-session
                on the KG(x;m) card.
              </span>
            </div>
          )}
          {isIE && (
            <div className="form-group">
              <label>Policy parameter — z_alpha (# std devs)</label>
              <input
                type="number" value={zAlphaStr} min={0} step="any"
                onChange={e => setZAlphaStr(e.target.value)}
                onBlur={() => setZAlphaStr(String(zAlpha))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                IE score = μ_n(x) − z_alpha · σ_n(x). z_alpha = 0 is pure
                exploitation (greedy); higher values pull toward under-observed
                θ. Typical exploration range: 0 – 3.
              </span>
            </div>
          )}

          {/* Advanced parameters (collapsible) */}
          <details style={{ marginBottom: 16, borderTop: '1px solid #e2e8f0', paddingTop: 12 }} open>
            <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: '#374151',
                              padding: '4px 0', userSelect: 'none' }}>
              Advanced parameters — belief prior, simulation truth, seed
            </summary>
            <div style={{ marginTop: 12, padding: '12px 16px', background: '#f8fafc',
                          borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 8 }}>
                Session
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label style={{ fontSize: 12 }}>Random seed</label>
                  <input type="number" value={seed} min={0} max={9999}
                    onChange={e => setSeed(Number(e.target.value))} />
                </div>
                <div className="form-group" style={{ justifyContent: 'flex-end' }}>
                  <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                    <input type="checkbox" checked={stationary}
                      onChange={e => setStationary(e.target.checked)}
                      style={{ width: 16, height: 16, cursor: 'pointer' }} />
                    Stationary regime (no market regime switching)
                  </label>
                </div>
              </div>

              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6,
                            marginTop: 16, marginBottom: 8 }}>
                Reporting
              </div>
              <div className="form-group">
                <label style={{ fontSize: 12 }}>Report level</label>
                <select value={reportLevel}
                        onChange={e => setReportLevel(e.target.value)}
                        style={{ maxWidth: 200 }}>
                  <option value="basic">Basic — core charts only</option>
                  <option value="advanced">Advanced — includes KG(x;m) S-curve diagnostic</option>
                </select>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  Basic shows the main KG, GP posterior, cash, and history cards.
                  Advanced adds diagnostic panels like the KG-vs-batch-size chart.
                </span>
              </div>

              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: 0.6,
                            marginTop: 16, marginBottom: 8 }}>
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
            Save and exit
          </button>
          <p style={{ fontSize: 11, color: '#94a3b8', margin: '8px 0 0 0', textAlign: 'center' }}>
            Saves your settings and returns to the landing page. Hit
            <b> Play the game</b> there to launch with these values.
          </p>
        </form>
      </div>
    </div>
  );
}
