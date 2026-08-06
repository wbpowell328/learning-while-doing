// Single sample path of deposits and redemptions over H trading days.
// Four series, all non-negative (deposits and redemptions rendered
// separately per Warren's spec): individual deposit, individual
// redemption, institutional deposit, institutional redemption.
// H is set in Advanced parameters (default 200 ≈ one working year).

const W = 720, H_svg = 300;
const PAD = { top: 40, right: 16, bottom: 40, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H_svg - PAD.top - PAD.bottom;

const SERIES = [
  { key: 'individual_deposit',       label: 'Individual — deposit',      color: '#2563eb', dash: null,  width: 1.6 },
  { key: 'individual_redemption',    label: 'Individual — redemption',   color: '#93c5fd', dash: '4,3', width: 1.6 },
  { key: 'institutional_deposit',    label: 'Institutional — deposit',   color: '#c2410c', dash: null,  width: 1.6 },
  { key: 'institutional_redemption', label: 'Institutional — redemption',color: '#fdba74', dash: '4,3', width: 1.6 },
];

function fmt(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(v / 1e3).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

export default function FlowSampleChart({ sample }) {
  if (!sample) {
    return (
      <div style={{ height: H_svg, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 14 }}>
        Loading sample path…
      </div>
    );
  }

  const { days, horizon } = sample;
  const allVals = SERIES.flatMap(s => sample[s.key] ?? []);
  const yMax = Math.max(1, ...allVals);
  const yLo  = 0;
  const yHi  = yMax * 1.10;

  const xS = (d) => PAD.left + ((d - 1) / Math.max(1, horizon - 1)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  // X ticks — five evenly-spaced days.
  const xTickCount = 5;
  const xTicks = Array.from({ length: xTickCount }, (_, i) =>
    1 + Math.round((i / (xTickCount - 1)) * (horizon - 1))
  );
  // Y ticks — five levels including 0 and yHi.
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));

  const linePath = (values) => days
    .map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d)},${yS(values[i] ?? 0)}`)
    .join('');

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H_svg}`} style={{ overflow: 'visible' }}>
      {/* Horizontal gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Series lines */}
      {SERIES.map(({ key, color, dash, width }) => (
        <path key={key} d={linePath(sample[key] ?? [])} fill="none"
              stroke={color} strokeWidth={width}
              strokeDasharray={dash ?? undefined}
              opacity={0.95} />
      ))}

      {/* X axis */}
      <line x1={PAD.left} x2={W - PAD.right} y1={H_svg - PAD.bottom} y2={H_svg - PAD.bottom}
            stroke="#94a3b8" />
      {xTicks.map((d) => (
        <g key={d}>
          <line x1={xS(d)} x2={xS(d)} y1={H_svg - PAD.bottom} y2={H_svg - PAD.bottom + 5}
                stroke="#94a3b8" />
          <text x={xS(d)} y={H_svg - PAD.bottom + 18}
                textAnchor="middle" fontSize={11} fill="#64748b">
            {d}
          </text>
        </g>
      ))}
      <text x={PAD.left + IW / 2} y={H_svg - 4}
            textAnchor="middle" fontSize={12} fill="#64748b">
        day
      </text>

      {/* Y axis */}
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H_svg - PAD.bottom}
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

      {/* Legend — two rows above the plot area to keep the plot clean. */}
      <g transform={`translate(${PAD.left},8)`}>
        {SERIES.slice(0, 2).map(({ label, color, dash }, i) => (
          <g key={i} transform={`translate(${i * 180},0)`}>
            <line x1={0} x2={22} y1={6} y2={6} stroke={color} strokeWidth={1.6}
                  strokeDasharray={dash ?? undefined} />
            <text x={28} y={9} fontSize={10} fill="#374151">{label}</text>
          </g>
        ))}
        <g transform="translate(0,18)">
          {SERIES.slice(2, 4).map(({ label, color, dash }, i) => (
            <g key={i} transform={`translate(${i * 180},0)`}>
              <line x1={0} x2={22} y1={6} y2={6} stroke={color} strokeWidth={1.6}
                    strokeDasharray={dash ?? undefined} />
              <text x={28} y={9} fontSize={10} fill="#374151">{label}</text>
            </g>
          ))}
        </g>
      </g>
    </svg>
  );
}
