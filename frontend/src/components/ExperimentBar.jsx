// Top-of-page experiment control bar. One row:
//
//   Starting point [θ]   (or [θ₁] [θ₂] for 2-D)
//   Run [N] days.
//   Then update using policy [policy▼]
//   with parameter [ρ].
//   Repeat [K] times.
//   {Run}
//
// One click executes an experiment on the backend: reset the belief,
// run iteration 1 at the user-typed θ for N days, then K more
// iterations picked by the chosen policy. Human policy uses K=0 (the
// UI forces it). Every Run resets the session state, so successive
// clicks give clean, independent comparisons.

import { useState, useEffect } from 'react';

const POLICY_OPTIONS_1D = [
  { value: 'kg',        label: 'KG offline correlated (analytic)' },
  { value: 'kg_indep',  label: 'KG offline independent' },
  { value: 'okg',       label: 'KG online correlated' },
  { value: 'okg_indep', label: 'KG online independent' },
  { value: 'ie',        label: 'IE — LCB' },
  { value: 'random',    label: 'Random' },
  { value: 'human',     label: 'Human — pick each θ' },
];
// Human is 1-D only for now.
const POLICY_OPTIONS_2D = POLICY_OPTIONS_1D.filter(o => o.value !== 'human');

// What kind of ρ parameter this policy takes (or null for none).
function paramForPolicy(policy) {
  if (policy === 'ie') return { key: 'z_alpha', label: 'z_alpha (# std devs)', default: 0, step: 'any', min: 0 };
  if (['kg', 'kg_indep', 'okg', 'okg_indep'].includes(policy)) {
    return { key: 'm_star', label: 'm* (days)', default: 1, step: 1, min: 1 };
  }
  return null;   // random, human — no parameter
}

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

export default function ExperimentBar({
  dim,               // 1 or 2
  defaultPolicy,     // seeds the dropdown from the SessionForm's Mode choice
  defaultMStar = 1,
  defaultZAlpha = 0,
  running = false,
  onRun,             // async (spec) => void — reset+K+1 iterations
  onOneMore,         // async (spec) => void — one more iteration, no reset
  canOneMore = false, // enable "One more" only after at least one Run
}) {
  const [theta1, setTheta1] = useState('');
  const [theta2, setTheta2] = useState('');
  const [nDays,  setNDays]  = useState('50');
  const [policy, setPolicy] = useState(defaultPolicy || 'kg');
  const [rho,    setRho]    = useState(String(paramForPolicy(defaultPolicy)?.default ?? ''));
  const [K,      setK]      = useState('0');

  // When the policy changes, seed ρ with that policy's default. Human
  // forces K=0.
  useEffect(() => {
    const meta = paramForPolicy(policy);
    if (meta) {
      if (policy === 'ie')      setRho(String(defaultZAlpha ?? meta.default));
      else                      setRho(String(defaultMStar  ?? meta.default));
    } else {
      setRho('');
    }
    if (policy === 'human') setK('0');
  }, [policy, defaultMStar, defaultZAlpha]);

  const paramMeta = paramForPolicy(policy);
  const isHuman = policy === 'human';
  const policyOptions = dim === 2 ? POLICY_OPTIONS_2D : POLICY_OPTIONS_1D;

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
      K: isHuman ? 0 : Math.max(0, Math.round(Number(K) || 0)),
      theta_init: dim === 2 ? [t1Num, t2Num] : t1Num,
    };
    if (paramMeta) {
      const v = Number(rho);
      if (Number.isFinite(v)) {
        spec[paramMeta.key] = paramMeta.key === 'm_star' ? Math.max(1, Math.round(v)) : v;
      }
    }
    onRun(spec);
  }

  // "One more" — step from the current session state using the bar's
  // current policy + ρ + N. No θ / K needed (θ picked by policy;
  // exactly one iteration executes). Only requires N to be valid.
  function commitOneMore() {
    if (running || !onOneMore || !nDaysValid) return;
    const spec = {
      n_days: Math.max(1, Math.round(nNum)),
      policy,   // let backend swap if it differs from current session policy
    };
    if (paramMeta) {
      const v = Number(rho);
      if (Number.isFinite(v)) {
        spec[paramMeta.key] = paramMeta.key === 'm_star' ? Math.max(1, Math.round(v)) : v;
      }
    }
    onOneMore(spec);
  }

  return (
    <div className="card" style={{
      display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
      padding: '12px 16px', marginBottom: 16,
    }}>
      <span style={labelStyle}>Starting point</span>
      {dim === 2 ? (
        <>
          <input type="number" step="any" value={theta1}
                 placeholder="θ₁" style={numStyle}
                 onChange={e => setTheta1(e.target.value)} />
          <input type="number" step="any" value={theta2}
                 placeholder="θ₂" style={numStyle}
                 onChange={e => setTheta2(e.target.value)} />
        </>
      ) : (
        <input type="number" step="any" value={theta1}
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
              onChange={e => setPolicy(e.target.value)}>
        {policyOptions.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>

      {paramMeta ? (
        <>
          <span style={labelStyle}>with parameter</span>
          <input type="number" step={paramMeta.step} min={paramMeta.min}
                 value={rho} placeholder="ρ" style={numStyleShort}
                 onChange={e => setRho(e.target.value)}
                 title={paramMeta.label} />
        </>
      ) : (
        <span style={{ ...labelStyle, fontStyle: 'italic', color: '#94a3b8' }}>
          (no parameter)
        </span>
      )}

      <span style={labelStyle}>. Repeat</span>
      <input type="number" min={0} step={1} value={K}
             placeholder="K" style={numStyleShort}
             disabled={isHuman}
             title={isHuman ? 'Human policy runs one iteration at a time (K=0)' : ''}
             onChange={e => setK(e.target.value)} />
      <span style={labelStyle}>times.</span>

      <button type="button"
              onClick={commit}
              disabled={running || !canRun}
              className="btn btn-primary"
              title={
                !thetaValid  ? 'Enter a starting θ' :
                !nDaysValid  ? 'Enter a positive N (days per iteration)' :
                'Reset the session and run K+1 iterations'
              }
              style={{ padding: '6px 20px', fontSize: 13 }}>
        {running ? 'running…' : 'Run'}
      </button>
      {/* Always rendered — disabled state communicates when it's usable.
          Conditional wrapping was hiding this button on some deploys. */}
      <button type="button"
              onClick={commitOneMore}
              disabled={running || !canOneMore || isHuman || !nDaysValid}
              className="btn btn-outline"
              title={
                isHuman
                  ? 'Human policy uses Run at each step'
                  : !canOneMore
                    ? 'Run an experiment first, then this steps one more iteration from where you left off'
                    : !nDaysValid
                      ? 'Enter a positive N (days per iteration)'
                      : 'Take one more iteration from the current state (no reset)'
              }
              style={{ padding: '6px 16px', fontSize: 13 }}>
        One more
      </button>
    </div>
  );
}
