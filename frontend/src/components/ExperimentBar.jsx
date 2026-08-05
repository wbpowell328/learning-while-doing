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

  function commit() {
    if (running || !onRun) return;
    const spec = {
      n_days: Math.max(1, Math.round(Number(nDays) || 1)),
      policy,
      K: isHuman ? 0 : Math.max(0, Math.round(Number(K) || 0)),
    };
    // θ
    if (dim === 2) {
      const t1 = Number(theta1), t2 = Number(theta2);
      if (!Number.isFinite(t1) || !Number.isFinite(t2)) return;
      spec.theta_init = [t1, t2];
    } else {
      const t1 = Number(theta1);
      if (!Number.isFinite(t1)) return;
      spec.theta_init = t1;
    }
    // ρ
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
  // exactly one iteration executes).
  function commitOneMore() {
    if (running || !onOneMore) return;
    const spec = {
      n_days: Math.max(1, Math.round(Number(nDays) || 1)),
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
             placeholder="N" style={numStyleShort}
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
              disabled={running}
              className="btn btn-primary"
              style={{ padding: '6px 20px', fontSize: 13 }}>
        {running ? 'running…' : 'Run'}
      </button>
      {onOneMore && (
        <button type="button"
                onClick={commitOneMore}
                disabled={running || !canOneMore || isHuman}
                className="btn btn-outline"
                title={
                  isHuman
                    ? 'Human policy uses Run at each step'
                    : !canOneMore
                      ? 'Run an experiment first, then this steps one more iteration from where you left off'
                      : 'Take one more iteration from the current state (no reset)'
                }
                style={{ padding: '6px 16px', fontSize: 13 }}>
          One more
        </button>
      )}
    </div>
  );
}
