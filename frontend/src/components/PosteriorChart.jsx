// Viewport + padding chosen to match KGChart's plot-area geometry so
// vertical marks at the same θ line up between the two charts stacked
// on top of each other (Warren's ask 2026-08). If you change KGChart's
// W / PAD.left / PAD.right, mirror it here.
const W = 720, H = 320;
const PAD = { top: 24, right: 96, bottom: 52, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const POLICY_COLOR = { random: '#2563eb', ie: '#d97706', kg: '#16a34a' };
const X_MIN = 0.01, X_MAX = 0.20;

function fmt(v) {
  const a = Math.abs(v);
  if (a >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  if (a < 10)  return `$${v.toFixed(2)}`;
  if (a < 100) return `$${v.toFixed(1)}`;
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

  const { impparams, std: stdRaw } = posterior;
  const color = POLICY_COLOR[policy] ?? '#2563eb';

  // Show the belief in $/day (its native scale), not per-batch totals, so
  // it lines up with the belief prior signal_std and the $/day reveal.
  // The backend scaled mean/std by n_days_used (per-batch $); divide it
  // back out, and divide the per-batch observation dots by the same
  // reference so the curve and the dots stay on one axis.
  const perDay = (Number.isFinite(Number(posterior.n_days_used)) && posterior.n_days_used > 0)
    ? Number(posterior.n_days_used) : 1;
  const mean = posterior.mean.map((m) => m / perDay);
  const std = stdRaw.map((s) => s / perDay);
  const dots = history.map(([c, cost]) => [c, cost / perDay]);

  // Y bounds — include band and observations
  const allY = [
    ...mean.map((m, i) => m + std[i]),
    ...mean.map((m, i) => m - std[i]),
    ...dots.map(([, cost]) => cost),
  ];
  const rawMin = Math.min(...allY);
  const rawMax = Math.max(...allY);
  const pad = (rawMax - rawMin) * 0.12 || Math.max(1, Math.abs(rawMax) * 0.1);
  const yLo = rawMin - pad;
  const yHi = rawMax + pad;

  const xS = (c) => PAD.left + ((c - X_MIN) / (X_MAX - X_MIN)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  // Confidence band polygon
  const upper = impparams.map((c, i) => `${xS(c)},${yS(mean[i] + std[i])}`);
  const lower = [...impparams].reverse().map((c, i, arr) => {
    const j = arr.length - 1 - i;
    return `${xS(c)},${yS(mean[j] - std[j])}`;
  });
  const bandPts = [...upper, ...lower].join(' ');

  // Mean line
  const meanPath = impparams
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c)},${yS(mean[i])}`)
    .join('');

  // Y ticks
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));
  const xTicks = [0.01, 0.05, 0.10, 0.15, 0.20];

  const best = posterior.best_impparam;

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

      {/* Best θ vertical */}
      <line x1={xS(best)} x2={xS(best)} y1={PAD.top} y2={H - PAD.bottom}
        stroke={color} strokeWidth={1.5} strokeDasharray="6,4" opacity={0.8} />
      <text x={xS(best) + 5} y={PAD.top + 14} fill={color} fontSize={11} fontWeight={600}>
        best θ={best.toFixed(3)}
      </text>

      {/* Observation dots (per-day, same as the GP curve) */}
      {dots.map(([c, cost], i) => (
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
        θ (cash buffer ratio)
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
        GP mean reward ($/day)
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
