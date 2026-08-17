const W = 680, H = 280;
const PAD = { top: 24, right: 24, bottom: 52, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;
const X_MIN = 0.01, X_MAX = 0.20;

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

function fmtDiff(v) {
  const sign = v >= 0 ? '+' : '';
  return `${sign}${fmt(v)}`;
}

export default function RevealPanel({ reveal }) {
  if (!reveal) return null;

  const {
    impparams, mean_reward,
    true_best_impparam, true_max_reward,
    player_best_impparam, player_best_reward,
    naive_reward, n_days_used,
  } = reveal;

  // "Total reward" is summed over one run of this many trading days (the
  // Horizon the user is running). Label the chart with it so the time
  // period behind "reward" and "/run" is explicit.
  const runDays = Number.isFinite(Number(n_days_used)) && n_days_used > 0
    ? Math.round(Number(n_days_used)) : null;
  const runLabel = runDays ? ` (${runDays} trading days)` : '';

  const allY = [...mean_reward, player_best_reward, naive_reward];
  const rawMin = Math.min(...allY);
  const rawMax = Math.max(...allY);
  const ygap = Math.max((rawMax - rawMin) * 0.15, 300);
  const yLo = rawMin - ygap;
  const yHi = rawMax + ygap;

  const xS = (c) => PAD.left + ((c - X_MIN) / (X_MAX - X_MIN)) * IW;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;

  const rewardPath = impparams
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c).toFixed(1)},${yS(mean_reward[i]).toFixed(1)}`)
    .join('');

  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));
  const xTicks = [0.01, 0.05, 0.10, 0.15, 0.20];

  // Gap: how much reward the player left on the table (true - player, ≥ 0).
  const playerGap = true_max_reward - player_best_reward;
  // vs naive: positive means the player earned more than naive.
  const vsNaive = player_best_reward - naive_reward;

  return (
    <div className="card reveal-panel">
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: '#1e293b' }}>
          Session complete — true reward curve revealed
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 3 }}>
          Estimated from {impparams.length} grid points × 50 independent simulations (held-out seed).
          {runDays ? ` Each run spans ${runDays} trading days; "reward" is the total over that run.` : ''}
        </div>
      </div>

      {/* Stats row */}
      <div className="reveal-stats">
        <div className="reveal-stat">
          <span className="reveal-stat-label">Your best θ</span>
          <span className="reveal-stat-value">{player_best_impparam.toFixed(3)}</span>
          <span className="reveal-stat-sub">~{fmt(player_best_reward)}/run</span>
        </div>
        <div className="reveal-stat">
          <span className="reveal-stat-label">True optimum</span>
          <span className="reveal-stat-value" style={{ color: '#dc2626' }}>
            {true_best_impparam.toFixed(3)}
          </span>
          <span className="reveal-stat-sub">~{fmt(true_max_reward)}/run</span>
        </div>
        <div className="reveal-stat">
          <span className="reveal-stat-label">Gap to optimum</span>
          <span className="reveal-stat-value" style={{ color: playerGap < 200 ? '#16a34a' : playerGap < 1000 ? '#d97706' : '#dc2626' }}>
            −{fmt(playerGap)}
          </span>
          <span className="reveal-stat-sub">per run</span>
        </div>
        <div className="reveal-stat">
          <span className="reveal-stat-label">vs. naive 10%</span>
          <span className="reveal-stat-value" style={{ color: vsNaive >= 0 ? '#16a34a' : '#dc2626' }}>
            {vsNaive >= 0 ? `+${fmt(vsNaive)}` : `−${fmt(-vsNaive)}`}
          </span>
          <span className="reveal-stat-sub">{fmt(naive_reward)}/run at 10%</span>
        </div>
      </div>

      {/* Chart */}
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible', marginTop: 12 }}>
        {/* Gridlines */}
        {yTicks.map((v, i) => (
          <line key={i} x1={PAD.left} x2={W - PAD.right}
            y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
        ))}

        {/* Naive 10% vertical */}
        <line x1={xS(0.10)} x2={xS(0.10)} y1={PAD.top} y2={H - PAD.bottom}
          stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.8} />

        {/* Player's best θ vertical */}
        <line x1={xS(player_best_impparam)} x2={xS(player_best_impparam)}
          y1={PAD.top} y2={H - PAD.bottom}
          stroke="#2563eb" strokeWidth={1.5} strokeDasharray="6,4" opacity={0.9} />

        {/* True best θ vertical */}
        <line x1={xS(true_best_impparam)} x2={xS(true_best_impparam)}
          y1={PAD.top} y2={H - PAD.bottom}
          stroke="#dc2626" strokeWidth={2} />

        {/* True reward curve */}
        <path d={rewardPath} fill="none" stroke="#374151" strokeWidth={2.5} />

        {/* Dot at true maximum */}
        <circle cx={xS(true_best_impparam)} cy={yS(true_max_reward)} r={5}
          fill="#dc2626" stroke="white" strokeWidth={1.5} />

        {/* Dot at player position */}
        <circle cx={xS(player_best_impparam)} cy={yS(player_best_reward)} r={5}
          fill="#2563eb" stroke="white" strokeWidth={1.5} />

        {/* X axis */}
        <line x1={PAD.left} x2={W - PAD.right}
          y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="#94a3b8" />
        {xTicks.map(v => (
          <g key={v}>
            <line x1={xS(v)} x2={xS(v)}
              y1={H - PAD.bottom} y2={H - PAD.bottom + 5} stroke="#94a3b8" />
            <text x={xS(v)} y={H - PAD.bottom + 18}
              textAnchor="middle" fontSize={11} fill="#64748b">{v.toFixed(2)}</text>
          </g>
        ))}
        <text x={PAD.left + IW / 2} y={H - 4}
          textAnchor="middle" fontSize={12} fill="#64748b">θ (cash buffer ratio)</text>

        {/* Y axis */}
        <line x1={PAD.left} x2={PAD.left}
          y1={PAD.top} y2={H - PAD.bottom} stroke="#94a3b8" />
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.left - 5} x2={PAD.left}
              y1={yS(v)} y2={yS(v)} stroke="#94a3b8" />
            <text x={PAD.left - 8} y={yS(v) + 4}
              textAnchor="end" fontSize={10} fill="#64748b">{fmt(v)}</text>
          </g>
        ))}
        <text
          transform={`translate(${PAD.left - 56},${PAD.top + IH / 2}) rotate(-90)`}
          textAnchor="middle" fontSize={12} fill="#64748b">Avg. total reward per run{runLabel}</text>

        {/* Legend */}
        <g transform={`translate(${W - PAD.right - 108},${PAD.top + 4})`}>
          <line x1={0} x2={16} y1={6} y2={6} stroke="#374151" strokeWidth={2.5} />
          <text x={20} y={10} fontSize={9} fill="#64748b">E[reward | θ]</text>
          <line x1={0} x2={16} y1={22} y2={22} stroke="#dc2626" strokeWidth={2} />
          <circle cx={8} cy={22} r={3} fill="#dc2626" stroke="white" strokeWidth={1} />
          <text x={20} y={26} fontSize={9} fill="#dc2626">true θ</text>
          <line x1={0} x2={16} y1={38} y2={38} stroke="#2563eb" strokeWidth={1.5} strokeDasharray="6,4" />
          <circle cx={8} cy={38} r={3} fill="#2563eb" stroke="white" strokeWidth={1} />
          <text x={20} y={42} fontSize={9} fill="#2563eb">your pick</text>
          <line x1={0} x2={16} y1={54} y2={54} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4,3" />
          <text x={20} y={58} fontSize={9} fill="#94a3b8">naive 10%</text>
        </g>
      </svg>
    </div>
  );
}
