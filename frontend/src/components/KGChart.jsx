// Line chart comparing three flavors of KG(x):
//   analytic correlated   (exact, FPD 2009)
//   MC correlated         (same estimand as analytic; MC noise diagnostic)
//   independent           (closed form; pretends zero cross-covariance)
//
// Purpose: pedagogical comparison — shows how independent-belief KG differs
// in shape from the correlated version, and how much noise the MC estimator
// carries relative to the analytic gold standard.

const W = 680, H = 320;
const PAD = { top: 24, right: 24, bottom: 56, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const X_MIN = 0.01, X_MAX = 0.20;

const SERIES = [
  { key: 'analytic_correlated', label: 'Correlated (analytic)', color: '#16a34a', dash: null },
  { key: 'mc_correlated',       label: 'Correlated (MC)',       color: '#7c3aed', dash: '5,3' },
  { key: 'independent',         label: 'Independent',           color: '#f59e0b', dash: null },
];

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) >= 10)   return `$${v.toFixed(0)}`;
  return `$${v.toFixed(2)}`;
}

export default function KGChart({ kg }) {
  if (!kg) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 14 }}>
        Create a session to see KG values.
      </div>
    );
  }

  const { c_stars, mc_samples } = kg;

  // Y bounds across all three series; include zero baseline for reference.
  const allY = SERIES.flatMap(s => kg[s.key]);
  const rawMax = Math.max(...allY, 1);
  const rawMin = Math.min(...allY, 0);
  const pad = (rawMax - rawMin) * 0.10 || 1;
  const yLo = Math.min(0, rawMin - pad);
  const yHi = rawMax + pad;

  const xS = (c) => PAD.left + ((c - X_MIN) / (X_MAX - X_MIN)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  const y0 = yS(0);
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));
  const xTicks = [0.01, 0.05, 0.10, 0.15, 0.20];

  const linePath = (series) => c_stars
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c)},${yS(series[i])}`)
    .join('');

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Zero baseline */}
      <line x1={PAD.left} x2={W - PAD.right} y1={y0} y2={y0}
            stroke="#94a3b8" strokeWidth={1} />

      {/* Series: line + markers */}
      {SERIES.map(({ key, color, dash }) => (
        <g key={key}>
          <path d={linePath(kg[key])} fill="none"
                stroke={color} strokeWidth={2}
                strokeDasharray={dash ?? undefined}
                opacity={0.9} />
          {c_stars.map((c, i) => (
            <circle key={i} cx={xS(c)} cy={yS(kg[key][i])} r={2.5}
                    fill={color} opacity={0.9} />
          ))}
        </g>
      ))}

      {/* X axis */}
      <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {xTicks.map((v) => (
        <g key={v}>
          <line x1={xS(v)} x2={xS(v)} y1={H - PAD.bottom} y2={H - PAD.bottom + 5}
                stroke="#94a3b8" />
          <text x={xS(v)} y={H - PAD.bottom + 18}
                textAnchor="middle" fontSize={11} fill="#64748b">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={PAD.left + IW / 2} y={H - 4}
            textAnchor="middle" fontSize={12} fill="#64748b">
        C* (cash buffer ratio)
      </text>

      {/* Y axis */}
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 5} x2={PAD.left} y1={yS(v)} y2={yS(v)} stroke="#94a3b8" />
          <text x={PAD.left - 8} y={yS(v) + 4}
                textAnchor="end" fontSize={10} fill="#64748b">
            {fmt(v)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(${PAD.left - 56},${PAD.top + IH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={12} fill="#64748b">
        KG(x)
      </text>

      {/* Legend */}
      <g transform={`translate(${W - PAD.right - 210},${PAD.top + 4})`}>
        {SERIES.map(({ label, color, dash }, i) => (
          <g key={i} transform={`translate(0,${i * 16})`}>
            <line x1={0} x2={22} y1={6} y2={6} stroke={color} strokeWidth={2}
                  strokeDasharray={dash ?? undefined} />
            <circle cx={11} cy={6} r={2.5} fill={color} />
            <text x={28} y={9} fontSize={10} fill="#374151">
              {label}{label.startsWith('Correlated (MC') ? `, n=${mc_samples}` : ''}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
