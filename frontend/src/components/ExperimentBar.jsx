// Top-of-page experiment control bar. One row:
//
//   Starting point [θ]   (or [θ₁] [θ₂] for 2-D)
//   Run [N] days.
//   Then update using policy [policy▼].
//   Repeat [K] times.
//   {Run}
//
// One click executes an experiment on the backend: reset the belief,
// run iteration 1 at the user-typed θ for N days, then K more
// iterations picked by the chosen policy. Human policy uses K=0 (the
// UI forces it). Every Run resets the session state, so successive
// clicks give clean, independent comparisons.
//
// The policy parameter (m* for KG, z_alpha for IE) is set once in the
// Advanced Parameters panel at session-create time; it lives on the
// session state and is *not* re-sent per Restart from this bar.

import { useState, useEffect } from 'react';

const POLICY_OPTIONS_1D = [
  { value: 'kg',                label: 'KG offline correlated (analytic)' },
  { value: 'kg_indep',          label: 'KG offline independent' },
  { value: 'okg',               label: 'KG online correlated (μ + KG(ρˡᵏʰᵈ))' },
  { value: 'okg_indep',         label: 'KG online independent' },
  { value: 'okg_ryzhov',        label: 'KG online correlated — Ryzhov (μ + (N−n)·KG(ρˡᵏʰᵈ))' },
  { value: 'ie',                label: 'IE' },
  { value: 'randomized_greedy', label: 'Randomized greedy' },
  { value: 'random',            label: 'Random' },
  { value: 'human',             label: 'Manual' },
];
// Manual (human) is 1-D only for now.
const POLICY_OPTIONS_2D = POLICY_OPTIONS_1D.filter(o => o.value !== 'human');

// Tight width defaults for the numeric fields — Warren's ask was to
// shrink Starting point, Run, Parameter, and Repeat to roughly a fifth
// of their previous size. Policy dropdown keeps its wider style since
// it holds long labels.
const numStyle = {
  padding: '4px 6px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 13, width: 55, boxSizing: 'border-box',
};
const numStyleShort = { ...numStyle, width: 45 };
const selectStyle = {
  padding: '4px 8px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 13, minWidth: 160,
};
const labelStyle = { color: '#475569', fontSize: 13 };

// Per-policy tunable-parameter descriptor. Returns null for policies
// that have no parameter (input area stays blank in the UI). Labels
// come from Warren's spec (2026-08); if a policy grows a second
// parameter later this schema will need to broaden.
function policyParamMeta(policy, values, handlers) {
  const { sessionMStar, sessionZAlpha, sessionSigmaGreedy } = values;
  const { onMStarChange, onZAlphaChange, onSigmaGreedyChange } = handlers;
  if (policy === 'okg' || policy === 'okg_indep' || policy === 'okg_ryzhov') {
    return {
      label: 'ρˡᵏʰᵈ',
      title: 'Lookahead parameter — multiplies the precision of the KG measurement noise',
      value: sessionMStar,
      step: 1, min: 1, integer: true,
      onCommit: onMStarChange,
    };
  }
  if (policy === 'ie') {
    return {
      label: 'θᴵᴱ',
      title: 'IE score = μ_n(θ) + θ^IE · σ_n(θ). Multiplies the std dev of μ^n_θ.',
      value: sessionZAlpha,
      step: 'any', min: 0, integer: false,
      onCommit: onZAlphaChange,
    };
  }
  if (policy === 'randomized_greedy') {
    return {
      label: 'ρˢᵗᵈᵈᵉᵛ',
      title: 'Std dev of Gaussian noise added to the greedy θ (same units as θ)',
      value: sessionSigmaGreedy,
      step: 'any', min: 0, integer: false,
      onCommit: onSigmaGreedyChange,
    };
  }
  return null;   // kg, kg_indep, random, human — no parameter
}

// Small controlled input for the current policy parameter. Commits on
// blur / Enter so we don't fire a request per keystroke; falls back
// to the last-good value if the user types garbage.
function PolicyParamInput({ meta }) {
  const { label, title, value, step, min, integer, onCommit } = meta;
  const [str, setStr] = useState(String(value));
  useEffect(() => { setStr(String(value)); }, [value]);
  const commit = () => {
    const v = integer ? Math.round(Number(str)) : Number(str);
    if (!Number.isFinite(v) || v < min) { setStr(String(value)); return; }
    if (v === value) return;
    if (onCommit) onCommit(v);
  };
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginLeft: 8 }}
          title={title}>
      <span style={labelStyle}>{label}</span>
      <input
        type="number" value={str} step={step} min={min}
        onChange={e => setStr(e.target.value)}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }}
        style={{ ...numStyleShort, width: 60 }}
      />
    </span>
  );
}

function formatDollars(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  const sign = v < 0 ? '−' : '';
  const abs = Math.abs(v);
  return `${sign}$${abs.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function ExperimentBar({
  dim,               // 1 or 2
  defaultPolicy,     // seeds the dropdown from the SessionForm's Mode choice
  running = false,
  onRun,             // async (spec) => void — reset+K+1 iterations
  onOneMore,         // async (spec) => void — one more iteration, no reset
  onRestart,         // async () => void — reset to initial conditions, do NOT run
  onManualEvaluate,  // async (theta, n_days) => void — Manual-policy single eval
  initialTheta,      // scalar (1-D) or [t1, t2] (2-D) — pre-fill for Test-point
  nextTheta,         // scalar (1-D) or [t1, t2] (2-D) — policy's preview of what
                     // it would pick next. When set, the Test-point box shows
                     // this instead of the last-tested θ. Null for Manual
                     // policy or fresh session, in which case we fall back to
                     // lastTheta / initialTheta as before.
  canOneMore = false, // enable "One more" only after at least one Run
  latestScore = null,     // total reward from the last batch (Restart or One more)
  cumulativeScore = null, // running total since the last Restart
  totalDays = 0,          // simulated days behind cumulativeScore; resets on Restart
  lastTheta = null,       // θ from the most recent iteration; null = fresh session
  // Current session values for the tunable policy parameters — one
  // and only one is shown at a time based on the selected policy.
  sessionMStar       = 1,
  sessionZAlpha      = 0,
  sessionSigmaGreedy = 0,
  onMStarChange,          // async (n) => void
  onZAlphaChange,         // async (x) => void
  onSigmaGreedyChange,    // async (x) => void
  onReveal,               // async () => void; null hides the Reveal button
  revealLoading = false,
  revealShown = false,    // true → Reveal already computed this session
  onOpenGameParams,       // () => void; null hides the button
}) {
  // Pre-fill θ from the Advanced-parameters "Initial value" field
  // (or 0.10 if the user didn't set one). 2-D case takes both dims.
  const _init1 = Array.isArray(initialTheta) ? Number(initialTheta[0]) : Number(initialTheta);
  const _init2 = Array.isArray(initialTheta) ? Number(initialTheta[1]) : Number(initialTheta);
  const [theta1, setTheta1] = useState(String(Number.isFinite(_init1) ? _init1 : 0.1));
  const [theta2, setTheta2] = useState(String(Number.isFinite(_init2) ? _init2 : 0.1));
  const [nDays,  setNDays]  = useState('50');
  const [policy, setPolicy] = useState(defaultPolicy || 'kg');
  const [K,      setK]      = useState('0');

  // Human forces K=0 (one iteration at a time). Handled at the source:
  // the policy dropdown's onChange snaps K to '0' when switching to
  // Human, and the Repeat input is disabled while Human is selected.
  const isHuman = policy === 'human';

  // Keep the θ input in sync with what actually happened. After every
  // Run / One more, App.jsx passes the last iteration's θ down; we
  // mirror it into the box (and swap the label to "Current point").
  // When history clears (Restart), lastTheta goes null and we snap
  // back to the pre-fill 0.1 so the box always reads well.
  useEffect(() => {
    // Priority: nextTheta (policy's preview of what it would pick next)
    // → lastTheta (last tested, fallback for Manual since it has no
    //   policy preview)
    // → initial θ from Advanced parameters (fresh session / Restart)
    const source = nextTheta != null ? nextTheta
                 : lastTheta != null ? lastTheta
                 : null;
    if (source == null) {
      setTheta1(String(Number.isFinite(_init1) ? _init1 : 0.1));
      setTheta2(String(Number.isFinite(_init2) ? _init2 : 0.1));
    } else if (dim === 2 && Array.isArray(source)) {
      setTheta1(String(source[0]));
      setTheta2(String(source[1]));
    } else {
      setTheta1(String(source));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nextTheta, lastTheta, dim, _init1, _init2]);

  // Label follows history: on a fresh / just-Restarted session the box
  // is the user's starting point; once anything has run, it's the
  // last-tested θ, so the label reads "Current point".
  // Always "Test point" — the θ that will be tested next when the
  // user hits Run (whether user-typed on first run / Manual, or the
  // policy's preview otherwise).
  const thetaLabel = 'Test point';
  const policyOptions = dim === 2 ? POLICY_OPTIONS_2D : POLICY_OPTIONS_1D;

  // Resolve the tunable parameter for the current policy — or null if
  // this policy has none, in which case the slot to the right of Restart
  // stays blank.
  const paramMeta = policyParamMeta(
    policy,
    { sessionMStar, sessionZAlpha, sessionSigmaGreedy },
    { onMStarChange, onZAlphaChange, onSigmaGreedyChange },
  );

  // Input validity — used to disable Run so the user can't fire off a
  // malformed request (empty θ silently becomes 0 via Number("")=0,
  // which is outside the [0.01, 0.20] box and used to crash to 500).
  const t1Str = String(theta1).trim();
  const t2Str = String(theta2).trim();
  const nStr  = String(nDays).trim();
  const t1Num = Number(t1Str), t2Num = Number(t2Str), nNum = Number(nStr);
  const theta1Valid = t1Str !== '' && Number.isFinite(t1Num);
  const theta2Valid = t2Str !== '' && Number.isFinite(t2Num);
  const nDaysValid  = nStr  !== '' && Number.isFinite(nNum) && nNum >= 1;
  const thetaValid  = dim === 2 ? (theta1Valid && theta2Valid) : theta1Valid;
  const canRun = thetaValid && nDaysValid;

  function commit() {
    if (running || !onRun || !canRun) return;
    const spec = {
      n_days: Math.max(1, Math.round(nNum)),
      policy,
      K: isHuman ? 0 : Math.min(20, Math.max(0, Math.round(Number(K) || 0))),
      theta_init: dim === 2 ? [t1Num, t2Num] : t1Num,
    };
    // m_star / z_alpha are session-level (set in Advanced parameters);
    // Restart intentionally does NOT override them per-request.
    onRun(spec);
  }

  // "One more" — step from the current session state using the bar's
  // current policy + N + Repeat. Repeat governs both Run and One
  // more symmetrically: K=0 means exactly one iteration; K>0 means
  // K+1 iterations, all policy-picked, no reset. θ is not sent (the
  // policy picks it from the current belief).
  function commitOneMore() {
    if (running || !onOneMore || !nDaysValid) return;
    const spec = {
      n_days: Math.max(1, Math.round(nNum)),
      policy,   // let backend swap if it differs from current session policy
      K: isHuman ? 0 : Math.min(20, Math.max(0, Math.round(Number(K) || 0))),
    };
    onOneMore(spec);
  }

  // Manual (human) policy: evaluate at the user-typed θ, one iteration,
  // no reset. Distinct from commitOneMore (which uses policy.propose)
  // because Manual wants the user's θ, not a policy pick.
  function commitManualEvaluate() {
    if (running || !onManualEvaluate || !canRun) return;
    const theta = dim === 2 ? [t1Num, t2Num] : t1Num;
    onManualEvaluate(theta, Math.max(1, Math.round(nNum)));
  }

  return (
    <div className="card" style={{
      display: 'flex', flexDirection: 'column', gap: 10,
      padding: '12px 16px', marginBottom: 16,
    }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <span style={labelStyle}>{thetaLabel}</span>
      {dim === 2 ? (
        <>
          <input type="number" step={0.01} min={0} value={theta1}
                 placeholder="θ₁" style={numStyle}
                 onChange={e => setTheta1(e.target.value)} />
          <input type="number" step={0.01} min={0} value={theta2}
                 placeholder="θ₂" style={numStyle}
                 onChange={e => setTheta2(e.target.value)} />
        </>
      ) : (
        <input type="number" step={0.01} min={0} value={theta1}
               placeholder="θ" style={numStyle}
               onChange={e => setTheta1(e.target.value)} />
      )}

      <span style={labelStyle}>Run</span>
      <input type="number" min={1} step={1} value={nDays}
             placeholder="N"
             style={{ ...numStyleShort, width: 65 }}
             onChange={e => setNDays(e.target.value)} />
      <span style={labelStyle}>days. Then update using policy</span>

      <select value={policy} style={selectStyle}
              onChange={e => {
                const p = e.target.value;
                setPolicy(p);
                if (p === 'human') setK('0');
              }}>
        {policyOptions.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>

      <span style={labelStyle}>. Repeat</span>
      <input type="number" min={0} max={20} step={1} value={K}
             placeholder="K" style={numStyleShort}
             disabled={isHuman}
             title={isHuman
               ? 'Human policy runs one iteration at a time (K=0)'
               : 'Number of policy-driven repetitions (0–20)'}
             onChange={e => setK(e.target.value)}
             onBlur={e => {
               // Snap the visible value into [0, 20] so a typo like
               // 200 shows as 20 instead of hiding a two-digit number
               // behind a narrow input. Backend was already clamping
               // on submit; this makes the enforcement visible.
               const v = Math.max(0, Math.min(20, Math.round(Number(e.target.value) || 0)));
               setK(String(v));
             }} />
      <span style={labelStyle}>times.</span>

      {/* Dual-mode button — three behaviours based on policy + history:
            Manual                     → always "Run", evaluates user's
                                         θ each click (no reset).
            Non-Manual, no history yet → "Run", full Restart-style
                                         experiment at user's θ.
            Non-Manual, has history    → "Repeat", K+1 more policy-
                                         picked iterations, no reset.
          Restart (to the right) is always a clean-slate reset. */}
      <button type="button"
              onClick={
                isHuman     ? commitManualEvaluate :
                canOneMore  ? commitOneMore :
                              commit
              }
              disabled={
                running ||
                (isHuman ? !canRun :
                  (canOneMore ? !nDaysValid : !canRun))
              }
              className="btn btn-primary"
              title={
                isHuman
                  ? (!canRun
                      ? 'Enter both θ and a positive N (days per iteration)'
                      : 'Evaluate the simulator at your θ (one iteration, no reset)')
                  : canOneMore
                    ? (!nDaysValid
                        ? 'Enter a positive N (days per iteration)'
                        : 'Repeat + 1 more iterations from the current state (no reset)')
                    : (!thetaValid
                        ? 'Enter a starting θ'
                        : !nDaysValid
                          ? 'Enter a positive N (days per iteration)'
                          : 'Run Repeat + 1 iterations from the starting θ')
              }
              style={{ padding: '6px 16px', fontSize: 13 }}>
        {running ? 'running…' : (isHuman ? 'Run' : (canOneMore ? 'Repeat' : 'Run'))}
      </button>
      <button type="button"
              onClick={() => { if (!running && onRestart) onRestart(); }}
              disabled={running || !onRestart}
              className="btn btn-primary"
              title="Reset to initial conditions (no simulation). The Run button will start a fresh experiment."
              style={{ padding: '6px 20px', fontSize: 13 }}>
        {running ? 'running…' : 'Restart'}
      </button>

      {/* Policy-parameter slot: label + editable value for the currently
          selected policy, or nothing when the policy has no parameter.
          Value lives on session state (mid-session edits propagate). */}
      {paramMeta && <PolicyParamInput meta={paramMeta} />}
    </div>

    {/* Second row: score readouts on the left, action buttons pushed
        to the right. Reveal truth and Game parameters live here (moved
        out of the deleted Stats card at Warren's request 2026-08). */}
    <div style={{ display: 'flex', gap: 12, alignItems: 'stretch', flexWrap: 'wrap' }}>
      <ScoreBox label="Total days"       value={totalDays} format="days"
                hint="Simulated days behind the cumulative score. Resets on Restart." />
      <ScoreBox label="Latest score"     value={latestScore}
                hint="Total reward from the last Restart or One more" />
      <ScoreBox label="Cumulative score" value={cumulativeScore}
                hint="Running total across all steps since the last Restart" />
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
        {onReveal && (
          <button type="button" onClick={onReveal}
                  disabled={running || revealLoading || revealShown}
                  className="btn btn-outline"
                  title={revealShown
                    ? 'Ground-truth reward curve already shown below'
                    : 'Plot the true underlying reward curve F(θ) to compare with your beliefs'}
                  style={{ padding: '6px 16px', fontSize: 13 }}>
            {revealLoading ? 'Computing…' : revealShown ? 'Truth revealed' : 'Reveal truth'}
          </button>
        )}
        {onOpenGameParams && (
          <button type="button" onClick={onOpenGameParams}
                  className="btn btn-outline"
                  title="Open the Game-parameters panel to change sim / belief settings. Save-and-exit returns you to the landing page."
                  style={{ padding: '6px 16px', fontSize: 13 }}>
            Game parameters
          </button>
        )}
      </div>
    </div>
    </div>
  );
}

function formatDays(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  return Math.round(v).toLocaleString();
}

function ScoreBox({ label, value, hint, format = 'dollars' }) {
  const shown = format === 'days' ? formatDays(value) : formatDollars(value);
  return (
    <div title={hint} style={{
      flex: '0 0 auto', minWidth: 160,
      border: '1px solid #e2e8f0', borderRadius: 6,
      padding: '6px 12px', background: '#f8fafc',
    }}>
      <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: '#0f172a',
                    fontVariantNumeric: 'tabular-nums' }}>
        {shown}
      </div>
    </div>
  );
}
