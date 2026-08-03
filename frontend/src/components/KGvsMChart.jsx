// KG(θ*; m) — how the information value at the "next θ" would grow if we
// repeated the same experiment m times (variance → σ²/m, precision → m·β).
//
// Motivates the S-curve effect: sometimes m=1 gives a tiny KG that the
// online-KG policy dismisses in favour of the μ_reward term, while a
// small batch of the same experiment would unlock a much larger KG.

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

export default function KGvsMChart({ data }) {
  if (!data || !data.m_values || data.m_values.length === 0) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 13 }}>
        Waiting for KG(m) curve…
      </div>
    );
  }

  const { theta, m_values, kg_values, noise_std, base_kg } = data;

  const xLo = m_values[0];
  const xHi = m_values[m_values.length - 1];
  const yMax = Math.max(...kg_values, base_kg, 1);
  const yLo = 0;
  const yHi = yMax * 1.10 || 1;

  const xS = (m) => PAD.left + ((m - xLo) / Math.max(xHi - xLo, 1)) * IW;
  const yS = (v) => PAD.top  + ((yHi - v) / (yHi - yLo)) * IH;

  const pathD = m_values
    .map((m, i) => `${i === 0 ? 'M' : 'L'}${xS(m).toFixed(1)},${yS(kg_values[i]).toFixed(1)}`)
    .join('');

  const yTicks = niceTicks(yLo, yHi, 5);
  const xTickCandidates = [1, 2, 5, 10, 20, 50, 100, 200];
  const xTicks = xTickCandidates.filter(t => t >= xLo && t <= xHi);
  if (xTicks[0] !== xLo) xTicks.unshift(xLo);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* KG(m) curve */}
      <path d={pathD} fill="none" stroke="#16a34a" strokeWidth={2.5} />

      {/* Marker at m=1 (what the main KG chart displays) */}
      <line x1={xS(1)} x2={xS(1)} y1={PAD.top} y2={H - PAD.bottom}
            stroke="#dc2626" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.85} />
      <circle cx={xS(1)} cy={yS(base_kg)} r={5}
              fill="#dc2626" stroke="white" strokeWidth={1.5} />
      <text x={xS(1) + 8} y={PAD.top + 12}
            fontSize={11} fontWeight={600} fill="#dc2626">
        m=1 (main chart)
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

      {/* Footer: β context */}
      <text x={W - PAD.right} y={H - 4} textAnchor="end"
            fontSize={10} fill="#94a3b8" fontStyle="italic">
        σ_ε = ${(noise_std/1000).toFixed(1)}k · precision β = 1/σ_ε² · effective precision at m = m·β
      </text>
    </svg>
  );
}
