// Grouped-bar chart comparing three flavors of KG at a coarse probe grid:
//   analytic correlated   (exact, FPD 2009)
//   MC correlated         (same estimand as analytic; MC noise diagnostic)
//   independent           (closed form; pretends zero cross-covariance)
//
// Purpose: pedagogical comparison — shows how independent-belief KG can
// mis-rank alternatives that a correlated GP prefers, and how much MC
// noise there is for the given sample count.

const W = 680, H = 300;
const PAD = { top: 24, right: 24, bottom: 56, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const COLOR = {
  analytic: '#16a34a',    // green — same as KG policy color
  mc:       '#7c3aed',    // violet
  indep:    '#f59e0b',    // amber
};

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

  const { c_stars, analytic_correlated, mc_correlated, independent, mc_samples } = kg;
  const n = c_stars.length;

  // Y bounds: KG is nonnegative in theory; include a bit of headroom + a
  // little slack for MC below zero.
  const allY = [...analytic_correlated, ...mc_correlated, ...independent];
  const rawMax = Math.max(...allY, 1);
  const rawMin = Math.min(...allY, 0);
  const pad = (rawMax - rawMin) * 0.10 || 1;
  const yLo = Math.min(0, rawMin - pad);
  const yHi = rawMax + pad;

  // X positions: one "slot" per probe; three bars per slot side-by-side.
  const slotW = IW / n;
  const barW = Math.min(24, (slotW * 0.7) / 3);
  const groupW = 3 * barW;

  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;
  const slotCenter = (i) => PAD.left + (i + 0.5) * slotW;
  const barX = (i, kIdx) => slotCenter(i) - groupW / 2 + kIdx * barW;

  const y0 = yS(0);
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));

  const bar = (i, kIdx, value, color) => {
    const x = barX(i, kIdx);
    const y = value >= 0 ? yS(value) : y0;
    const h = Math.max(0.5, Math.abs(yS(value) - y0));
    return <rect key={`${i}-${kIdx}`} x={x} y={y} width={barW - 1} height={h}
                 fill={color} fillOpacity={0.85} />;
  };

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Horizontal gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Zero baseline (bold) */}
      <line x1={PAD.left} x2={W - PAD.right} y1={y0} y2={y0}
            stroke="#94a3b8" strokeWidth={1} />

      {/* Bars */}
      {c_stars.map((_, i) => (
        <g key={i}>
          {bar(i, 0, analytic_correlated[i], COLOR.analytic)}
          {bar(i, 1, mc_correlated[i],       COLOR.mc)}
          {bar(i, 2, independent[i],         COLOR.indep)}
        </g>
      ))}

      {/* X axis: probe labels */}
      <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {c_stars.map((c, i) => (
        <text key={i} x={slotCenter(i)} y={H - PAD.bottom + 16}
              textAnchor="middle" fontSize={11} fill="#64748b">
          {c.toFixed(2)}
        </text>
      ))}
      <text x={PAD.left + IW / 2} y={H - 4}
            textAnchor="middle" fontSize={12} fill="#64748b">
        C* probe points (cash buffer ratio)
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
        KG value
      </text>

      {/* Legend */}
      <g transform={`translate(${PAD.left + 8},${PAD.top + 4})`}>
        <rect x={0}  y={0}  width={10} height={10} fill={COLOR.analytic} fillOpacity={0.85} />
        <text x={14} y={9}  fontSize={10} fill="#374151">Correlated (analytic)</text>
        <rect x={0}  y={16} width={10} height={10} fill={COLOR.mc} fillOpacity={0.85} />
        <text x={14} y={25} fontSize={10} fill="#374151">Correlated (MC, n={mc_samples})</text>
        <rect x={0}  y={32} width={10} height={10} fill={COLOR.indep} fillOpacity={0.85} />
        <text x={14} y={41} fontSize={10} fill="#374151">Independent (closed form)</text>
      </g>
    </svg>
  );
}
