const W = 680, H = 320;
const PAD = { top: 24, right: 24, bottom: 52, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const POLICY_COLOR = { random: '#2563eb', ie: '#d97706', kg: '#16a34a' };
const X_MIN = 0.01, X_MAX = 0.20;

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

export default function PosteriorChart({ posterior, history, policy }) {
  if (!posterior) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: 14 }}>
        Create a session to see the GP posterior.
      </div>
    );
  }

  const { c_stars, mean, std } = posterior;
  const color = POLICY_COLOR[policy] ?? '#2563eb';

  // Y bounds — include band and observations
  const allY = [
    ...mean.map((m, i) => m + std[i]),
    ...mean.map((m, i) => m - std[i]),
    ...history.map(([, cost]) => cost),
  ];
  const rawMin = Math.min(...allY);
  const rawMax = Math.max(...allY);
  const pad = (rawMax - rawMin) * 0.12 || 1000;
  const yLo = rawMin - pad;
  const yHi = rawMax + pad;

  const xS = (c) => PAD.left + ((c - X_MIN) / (X_MAX - X_MIN)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  // Confidence band polygon
  const upper = c_stars.map((c, i) => `${xS(c)},${yS(mean[i] + std[i])}`);
  const lower = [...c_stars].reverse().map((c, i, arr) => {
    const j = arr.length - 1 - i;
    return `${xS(c)},${yS(mean[j] - std[j])}`;
  });
  const bandPts = [...upper, ...lower].join(' ');

  // Mean line
  const meanPath = c_stars
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c)},${yS(mean[i])}`)
    .join('');

  // Y ticks
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));
  const xTicks = [0.01, 0.05, 0.10, 0.15, 0.20];

  const best = posterior.best_c_star;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
          y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Band */}
      <polygon points={bandPts} fill={color} fillOpacity={0.15} />

      {/* Mean line */}
      <path d={meanPath} fill="none" stroke={color} strokeWidth={2.5} />

      {/* Best C* vertical */}
      <line x1={xS(best)} x2={xS(best)} y1={PAD.top} y2={H - PAD.bottom}
        stroke={color} strokeWidth={1.5} strokeDasharray="6,4" opacity={0.8} />
      <text x={xS(best) + 5} y={PAD.top + 14} fill={color} fontSize={11} fontWeight={600}>
        best C*={best.toFixed(3)}
      </text>

      {/* Observation dots */}
      {history.map(([c, cost], i) => (
        <circle key={i} cx={xS(c)} cy={yS(cost)} r={5}
          fill={color} stroke="white" strokeWidth={1.5} opacity={0.9} />
      ))}

      {/* X axis */}
      <line x1={PAD.left} x2={W - PAD.right}
        y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="#94a3b8" />
      {xTicks.map((v) => (
        <g key={v}>
          <line x1={xS(v)} x2={xS(v)}
            y1={H - PAD.bottom} y2={H - PAD.bottom + 5} stroke="#94a3b8" />
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
      <line x1={PAD.left} x2={PAD.left}
        y1={PAD.top} y2={H - PAD.bottom} stroke="#94a3b8" />
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 5} x2={PAD.left}
            y1={yS(v)} y2={yS(v)} stroke="#94a3b8" />
          <text x={PAD.left - 8} y={yS(v) + 4}
            textAnchor="end" fontSize={10} fill="#64748b">
            {fmt(v)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(${PAD.left - 56},${PAD.top + IH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={12} fill="#64748b">
        Total cost
      </text>

      {/* Legend */}
      <g transform={`translate(${W - PAD.right - 130},${PAD.top + 4})`}>
        <rect width={8} height={8} y={1} rx={1} fill={color} fillOpacity={0.25} stroke="none" />
        <text x={12} y={9} fontSize={10} fill="#64748b">GP ±1σ</text>
        <line x1={0} x2={8} y1={20} y2={20} stroke={color} strokeWidth={2.5} />
        <text x={12} y={23} fontSize={10} fill="#64748b">GP mean</text>
        <circle cx={4} cy={35} r={4} fill={color} stroke="white" strokeWidth={1.5} />
        <text x={12} y={38} fontSize={10} fill="#64748b">Observation</text>
      </g>
    </svg>
  );
}
