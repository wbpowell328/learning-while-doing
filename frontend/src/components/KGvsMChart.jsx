// KG(θ*; m) — how the information value at the "next θ" would grow if we
// repeated the same experiment m times (variance → σ²/m, precision → m·β).
//
// Motivates the S-curve effect: sometimes m=1 gives a tiny KG that the
// online-KG policy dismisses in favour of the μ_reward term, while a
// small batch of the same experiment would unlock a much larger KG.

import { useState, useEffect } from 'react';

const W = 640, H = 260;
const PAD = { top: 20, right: 24, bottom: 48, left: 68 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

function fmt$(v) {
  const a = Math.abs(v);
  if (a >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  if (a >= 10)   return `$${v.toFixed(0)}`;
  return `$${v.toFixed(2)}`;
}

function niceTicks(lo, hi, n = 5) {
  return Array.from({ length: n }, (_, i) => lo + (i / (n - 1)) * (hi - lo));
}

export default function KGvsMChart({
  data, onSigmaEpsChange, sigmaEpsPending = false,
  mMax = 50, onMMaxChange,
  onThetaChange,
}) {
  // Local text state so mid-typing doesn't fire a request per keystroke.
  const [sigmaStr, setSigmaStr] = useState('');
  const [mMaxStr,  setMMaxStr]  = useState(String(mMax));
  const [thetaStr, setThetaStr] = useState('');
  useEffect(() => {
    if (data?.noise_std != null) setSigmaStr(String(Math.round(data.noise_std)));
  }, [data?.noise_std]);
  useEffect(() => { setMMaxStr(String(mMax)); }, [mMax]);
  useEffect(() => {
    if (data?.theta != null) setThetaStr(data.theta.toFixed(3));
  }, [data?.theta]);

  if (!data || !data.m_values || data.m_values.length === 0) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 13 }}>
        Waiting for KG(m) curve…
      </div>
    );
  }

  const { theta, m_values, kg_values, noise_std, noise_std_belief, base_kg } = data;
  // kg_values is the CORRELATED KG curve (what the policy uses); the
  // INDEPENDENT-scalar curve is served alongside for pedagogical
  // comparison and exhibits the classical S when the noise regime lines
  // up. Both are on the same y-scale.
  const kg_values_indep = data.kg_values_independent ?? [];
  // Diagnostic readouts — control whether the S is visible or not.
  const delta_corr        = data.delta_corr;
  const sigma_tilde_corr  = data.sigma_tilde_corr;
  const delta_indep       = data.delta_indep;
  const sigma_tilde_indep = data.sigma_tilde_indep;
  const zscore_corr  = (delta_corr  != null && sigma_tilde_corr  > 1e-12) ? delta_corr  / sigma_tilde_corr  : null;
  const zscore_indep = (delta_indep != null && sigma_tilde_indep > 1e-12) ? delta_indep / sigma_tilde_indep : null;
  const isOverridden = noise_std_belief != null &&
                       Math.abs(noise_std - noise_std_belief) > 1e-6;

  const commit = () => {
    if (!onSigmaEpsChange) return;
    const v = Number(sigmaStr);
    if (!Number.isFinite(v) || v <= 0) {
      setSigmaStr(String(Math.round(noise_std)));
      return;
    }
    if (Math.abs(v - noise_std) < 1e-6) return;   // no change
    onSigmaEpsChange(v);
  };
  const resetToBelief = () => {
    if (!onSigmaEpsChange || noise_std_belief == null) return;
    setSigmaStr(String(Math.round(noise_std_belief)));
    onSigmaEpsChange(noise_std_belief);
  };
  const commitMMax = () => {
    if (!onMMaxChange) return;
    const v = Math.round(Number(mMaxStr));
    if (!Number.isFinite(v) || v < 2) { setMMaxStr(String(mMax)); return; }
    const clamped = Math.max(2, Math.min(v, 5000));
    if (clamped === mMax) return;
    onMMaxChange(clamped);
  };
  const commitTheta = () => {
    if (!onThetaChange) return;
    const v = Number(thetaStr);
    if (!Number.isFinite(v) || v <= 0) { setThetaStr(theta.toFixed(3)); return; }
    if (Math.abs(v - theta) < 1e-6) return;
    onThetaChange(v);
  };
  const resetThetaToArgmaxKG = () => {
    if (!onThetaChange) return;
    setThetaStr('');   // sending null makes the backend pick argmax(offline_KG)
    onThetaChange(null);
  };

  const xLo = m_values[0];
  const xHi = m_values[m_values.length - 1];
  const yMax = Math.max(...kg_values, ...kg_values_indep, base_kg, 1);
  const yLo = 0;
  const yHi = yMax * 1.10 || 1;

  const xS = (m) => PAD.left + ((m - xLo) / Math.max(xHi - xLo, 1)) * IW;
  const yS = (v) => PAD.top  + ((yHi - v) / (yHi - yLo)) * IH;

  const pathD = m_values
    .map((m, i) => `${i === 0 ? 'M' : 'L'}${xS(m).toFixed(1)},${yS(kg_values[i]).toFixed(1)}`)
    .join('');
  const pathDIndep = kg_values_indep.length === m_values.length
    ? m_values
        .map((m, i) => `${i === 0 ? 'M' : 'L'}${xS(m).toFixed(1)},${yS(kg_values_indep[i]).toFixed(1)}`)
        .join('')
    : '';

  const yTicks = niceTicks(yLo, yHi, 5);
  const xTickCandidates = [1, 2, 5, 10, 20, 50, 100, 200];
  const xTicks = xTickCandidates.filter(t => t >= xLo && t <= xHi);
  if (xTicks[0] !== xLo) xTicks.unshift(xLo);

  return (
    <div>
      {/* σ_ε control — recomputes the KG(m) curve in-place using the
          override, without disturbing the current session's belief. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10,
                    fontSize: 12, color: '#475569', marginBottom: 8, flexWrap: 'wrap' }}>
        <label style={{ fontWeight: 600 }}>σ_ε (per-run reward noise):</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span>$</span>
          <input
            type="number"
            value={sigmaStr}
            min={1}
            step="any"
            onChange={e => setSigmaStr(e.target.value)}
            onBlur={commit}
            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }}
            disabled={sigmaEpsPending}
            style={{ width: 90, padding: '3px 6px', border: '1px solid #cbd5e1',
                     borderRadius: 4, fontSize: 12 }}
          />
        </div>
        <label style={{ fontWeight: 600, marginLeft: 8 }}>m<sub>max</sub>:</label>
        <input
          type="number"
          value={mMaxStr}
          min={2}
          max={5000}
          step={1}
          onChange={e => setMMaxStr(e.target.value)}
          onBlur={commitMMax}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commitMMax(); } }}
          disabled={sigmaEpsPending}
          style={{ width: 70, padding: '3px 6px', border: '1px solid #cbd5e1',
                   borderRadius: 4, fontSize: 12 }}
        />
        <label style={{ fontWeight: 600, marginLeft: 8 }}>θ:</label>
        <input
          type="number"
          value={thetaStr}
          step="any"
          onChange={e => setThetaStr(e.target.value)}
          onBlur={commitTheta}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commitTheta(); } }}
          disabled={sigmaEpsPending}
          style={{ width: 80, padding: '3px 6px', border: '1px solid #cbd5e1',
                   borderRadius: 4, fontSize: 12 }}
        />
        <button
          type="button"
          onClick={resetThetaToArgmaxKG}
          disabled={sigmaEpsPending}
          title="Reset θ to argmax(offline KG) — the θ the KG policy would sample next"
          style={{ background: 'transparent', border: '1px solid #cbd5e1',
                   borderRadius: 4, padding: '2px 8px', fontSize: 11,
                   color: '#475569', cursor: 'pointer' }}>
          θ = argmax KG
        </button>
        {sigmaEpsPending && <span style={{ color: '#94a3b8' }}>recomputing…</span>}
        {isOverridden && !sigmaEpsPending && (
          <>
            <span style={{ color: '#b45309', fontStyle: 'italic' }}>
              what-if refit — belief's σ_ε = ${Math.round(noise_std_belief).toLocaleString()}
            </span>
            <button
              type="button"
              onClick={resetToBelief}
              style={{ background: 'transparent', border: '1px solid #cbd5e1',
                       borderRadius: 4, padding: '2px 8px', fontSize: 11,
                       color: '#475569', cursor: 'pointer' }}>
              reset
            </button>
          </>
        )}
      </div>

      {/* Diagnostic row: |Δ|, σ̃(m=1) and their ratio for both formulations.
          The S is visible only when Δ/σ̃ ≳ 2 or so — that's the regime where
          f(-z) is near-zero, forcing KG(m=1) small. Values in belief-frame
          $ (posterior means at θ vs current best, and Bayesian mean-shift
          std at a single observation). */}
      {(delta_corr != null || delta_indep != null) && (
        <div style={{ display: 'grid',
                      gridTemplateColumns: '110px 1fr 1fr 1fr',
                      gap: '4px 12px',
                      fontSize: 11, color: '#475569',
                      background: '#f8fafc',
                      padding: '6px 10px', borderRadius: 4,
                      marginBottom: 8 }}>
          <div style={{ fontWeight: 600 }}></div>
          <div style={{ fontWeight: 600 }}>|Δ|</div>
          <div style={{ fontWeight: 600 }}>σ̃(m=1)</div>
          <div style={{ fontWeight: 600 }}>Δ/σ̃</div>

          <div style={{ color: '#16a34a', fontWeight: 600 }}>Correlated</div>
          <div>${delta_corr?.toFixed(1)}</div>
          <div>${sigma_tilde_corr?.toFixed(1)}</div>
          <div style={{ color: (zscore_corr != null && zscore_corr > 2) ? '#dc2626' : '#475569' }}>
            {zscore_corr?.toFixed(2)}
          </div>

          <div style={{ color: '#2563eb', fontWeight: 600 }}>Independent</div>
          <div>${delta_indep?.toFixed(1)}</div>
          <div>${sigma_tilde_indep?.toFixed(1)}</div>
          <div style={{ color: (zscore_indep != null && zscore_indep > 2) ? '#dc2626' : '#475569' }}>
            {zscore_indep?.toFixed(2)}
          </div>

          <div style={{ gridColumn: '1 / -1', color: '#94a3b8', fontStyle: 'italic', marginTop: 2 }}>
            The classical S needs Δ/σ̃ ≳ 2 (red) — that's where f(−z) → 0 and KG(m=1) collapses.
            Smaller ratios give the concave-plateau shape.
          </div>
        </div>
      )}

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Correlated KG(m) curve — what the KG policy actually uses. */}
      <path d={pathD} fill="none" stroke="#16a34a" strokeWidth={2.5} />

      {/* Independent-scalar KG(m) — pedagogically-classical S-curve view. */}
      {pathDIndep && (
        <path d={pathDIndep} fill="none" stroke="#2563eb" strokeWidth={2}
              strokeDasharray="5,3" opacity={0.9} />
      )}

      {/* Marker at m=1. When the belief is not overridden, this value is
          exactly what the main KG chart shows at θ*. When the σ_ε override
          is active it's the m=1 value under the what-if refit, so the
          label drops the "main chart" reference to avoid confusion. */}
      <line x1={xS(1)} x2={xS(1)} y1={PAD.top} y2={H - PAD.bottom}
            stroke="#dc2626" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.85} />
      <circle cx={xS(1)} cy={yS(base_kg)} r={5}
              fill="#dc2626" stroke="white" strokeWidth={1.5} />
      <text x={xS(1) + 8} y={PAD.top + 12}
            fontSize={11} fontWeight={600} fill="#dc2626">
        {isOverridden ? 'm=1 (single experiment)' : 'm=1 (main chart)'}
      </text>

      {/* X axis */}
      <line x1={PAD.left} x2={W - PAD.right}
            y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="#94a3b8" />
      {xTicks.map(v => (
        <g key={v}>
          <line x1={xS(v)} x2={xS(v)}
                y1={H - PAD.bottom} y2={H - PAD.bottom + 5} stroke="#94a3b8" />
          <text x={xS(v)} y={H - PAD.bottom + 18}
                textAnchor="middle" fontSize={11} fill="#64748b">{v}</text>
        </g>
      ))}
      <text x={PAD.left + IW / 2} y={H - 4}
            textAnchor="middle" fontSize={12} fill="#64748b">
        m — number of repeat experiments at θ = {theta.toFixed(3)}
      </text>

      {/* Y axis */}
      <line x1={PAD.left} x2={PAD.left}
            y1={PAD.top} y2={H - PAD.bottom} stroke="#94a3b8" />
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 5} x2={PAD.left}
                y1={yS(v)} y2={yS(v)} stroke="#94a3b8" />
          <text x={PAD.left - 8} y={yS(v) + 4}
                textAnchor="end" fontSize={10} fill="#64748b">{fmt$(v)}</text>
        </g>
      ))}
      <text
        transform={`translate(${PAD.left - 52},${PAD.top + IH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={12} fill="#64748b">
        KG(θ; m)
      </text>

      {/* Legend — two lines: correlated (policy) vs independent (classical) */}
      <g transform={`translate(${W - PAD.right - 210},${PAD.top + 4})`}>
        <line x1={0} x2={20} y1={6} y2={6} stroke="#16a34a" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fill="#374151">
          Correlated KG (what the policy uses)
        </text>
        <line x1={0} x2={20} y1={22} y2={22} stroke="#2563eb" strokeWidth={2}
              strokeDasharray="5,3" opacity={0.9} />
        <text x={26} y={26} fontSize={10} fill="#374151">
          Independent beliefs (no cross-updates)
        </text>
      </g>

      {/* Footer: β context */}
      <text x={W - PAD.right} y={H - 4} textAnchor="end"
            fontSize={10} fill="#94a3b8" fontStyle="italic">
        σ_ε = ${(noise_std/1000).toFixed(1)}k · precision β = 1/σ_ε² · effective precision at m = m·β
      </text>
      </svg>
    </div>
  );
}
