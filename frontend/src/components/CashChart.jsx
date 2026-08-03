const W = 680, H = 240;
const PAD = { top: 16, right: 48, bottom: 36, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

function fmt(v) {
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (abs >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

export default function CashChart({ result }) {
  if (!result) return null;
  const { cash_series, event_log, impparam, initial_aum, days } = result;

  const target = impparam * initial_aum;
  const cashMin = Math.min(...cash_series);
  const cashMax = Math.max(...cash_series);
  const rawMin = Math.min(cashMin, 0);
  const rawMax = Math.max(cashMax, target);
  const ygap = Math.max((rawMax - rawMin) * 0.15, 2000);
  const yLo = rawMin - ygap;
  const yHi = rawMax + ygap;

  const xS = (day) => PAD.left + (day / Math.max(days - 1, 1)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  const zeroY = yS(0);
  const targetY = yS(target);

  const cashPath = cash_series
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${xS(i).toFixed(1)},${yS(v).toFixed(1)}`)
    .join('');

  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));

  const totalWeeks = Math.round(days / 5);
  const wStep = totalWeeks <= 13 ? 2 : 4;
  const weekTicks = [];
  for (let w = 0; w <= totalWeeks; w += wStep) weekTicks.push(w);

  const hasShortfall = cashMin < 0;

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 8, fontSize: 11, color: '#64748b', flexWrap: 'wrap' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke="#2563eb" strokeWidth="2" /></svg>
          Cash balance
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke="#475569" strokeWidth="1.5" strokeDasharray="5,3" /></svg>
          Target ({(impparam * 100).toFixed(1)}% of AUM = {fmt(target)})
        </span>
        {event_log.length > 0 && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <svg width="4" height="12"><line x1="2" y1="0" x2="2" y2="12" stroke="#dc2626" strokeWidth="2" /></svg>
            Redemption
            <svg width="4" height="12"><line x1="2" y1="0" x2="2" y2="12" stroke="#16a34a" strokeWidth="2" /></svg>
            Subscription
          </span>
        )}
        {hasShortfall && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 12, height: 10, background: '#fee2e2', display: 'inline-block', borderRadius: 2 }} />
            Shortfall zone
          </span>
        )}
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
        {/* Y gridlines */}
        {yTicks.map((v, i) => (
          <line key={i} x1={PAD.left} x2={W - PAD.right}
            y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
        ))}

        {/* Shortfall zone */}
        {hasShortfall && zeroY >= PAD.top && zeroY <= H - PAD.bottom && (
          <rect x={PAD.left} y={zeroY} width={IW}
            height={H - PAD.bottom - zeroY}
            fill="#fee2e2" opacity={0.5} />
        )}

        {/* Zero line */}
        <line x1={PAD.left} x2={W - PAD.right} y1={zeroY} y2={zeroY}
          stroke="#94a3b8" strokeWidth={1} />

        {/* Target line */}
        <line x1={PAD.left} x2={W - PAD.right} y1={targetY} y2={targetY}
          stroke="#475569" strokeWidth={1.5} strokeDasharray="6,4" />
        <text x={W - PAD.right + 4} y={targetY + 4} fontSize={9} fill="#475569">
          {(impparam * 100).toFixed(0)}%
        </text>

        {/* Jump event markers */}
        {event_log.map((e, i) => (
          <line key={i}
            x1={xS(e.day)} x2={xS(e.day)}
            y1={H - PAD.bottom - 10} y2={H - PAD.bottom + 3}
            stroke={e.direction === -1 ? '#dc2626' : '#16a34a'}
            strokeWidth={1.5}
          />
        ))}

        {/* Cash line */}
        <path d={cashPath} fill="none" stroke="#2563eb" strokeWidth={1.8} />

        {/* X axis */}
        <line x1={PAD.left} x2={W - PAD.right}
          y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="#94a3b8" />
        {weekTicks.map(w => {
          const day = Math.min(w * 5, days - 1);
          return (
            <g key={w}>
              <line x1={xS(day)} x2={xS(day)}
                y1={H - PAD.bottom} y2={H - PAD.bottom + 4} stroke="#94a3b8" />
              <text x={xS(day)} y={H - PAD.bottom + 15}
                textAnchor="middle" fontSize={10} fill="#64748b">W{w}</text>
            </g>
          );
        })}
        <text x={PAD.left + IW / 2} y={H - 2}
          textAnchor="middle" fontSize={11} fill="#64748b">Week</text>

        {/* Y axis */}
        <line x1={PAD.left} x2={PAD.left}
          y1={PAD.top} y2={H - PAD.bottom} stroke="#94a3b8" />
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.left - 4} x2={PAD.left}
              y1={yS(v)} y2={yS(v)} stroke="#94a3b8" />
            <text x={PAD.left - 6} y={yS(v) + 4}
              textAnchor="end" fontSize={9} fill="#64748b">{fmt(v)}</text>
          </g>
        ))}
        <text
          transform={`translate(${PAD.left - 52},${PAD.top + IH / 2}) rotate(-90)`}
          textAnchor="middle" fontSize={11} fill="#64748b">Cash ($)</text>
      </svg>
    </div>
  );
}
