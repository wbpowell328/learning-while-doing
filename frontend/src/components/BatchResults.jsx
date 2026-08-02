// Batch benchmarking results view.
//
// Two panels:
//   1. Line/bar chart of mean terminal cost by policy (with std-dev error bars),
//      plus a horizontal dashed line marking the ground-truth minimum.
//   2. Table of per-policy aggregates.

const W = 720, H = 340;
const PAD = { top: 24, right: 24, bottom: 76, left: 84 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(2)}k`;
  return `$${v.toFixed(1)}`;
}

export default function BatchResults({ batch, onReset }) {
  if (!batch) return null;
  const { family, sims_per_policy, budget, session_seed,
          true_best_c_star, true_min_cost, policies } = batch;

  // ------------------------------------------------------------------
  // Chart: mean terminal cost per policy with ±1σ error bars
  // ------------------------------------------------------------------
  const yVals = policies.flatMap(p => [
    p.mean_terminal_cost + p.std_terminal_cost,
    p.mean_terminal_cost - p.std_terminal_cost,
  ]);
  yVals.push(true_min_cost);
  const rawMax = Math.max(...yVals);
  const rawMin = Math.min(...yVals);
  const pad = (rawMax - rawMin) * 0.12 || 1;
  const yLo = Math.max(0, rawMin - pad);
  const yHi = rawMax + pad;
  const yS = (v) => PAD.top + ((yHi - v) / (yHi - yLo)) * IH;
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (i / 4) * (yHi - yLo));

  const n = policies.length;
  const slotW = IW / n;
  const barW = Math.min(28, slotW * 0.55);
  const barX = (i) => PAD.left + (i + 0.5) * slotW - barW / 2;

  // KG family colors by variant, IE family a single color
  const isIE = family === 'IE';
  const kgColorMap = [ '#16a34a', '#7c3aed', '#f59e0b', '#14532d', '#78350f' ];
  const barColor = (i, p) => {
    if (isIE) return '#2563eb';
    return kgColorMap[i] ?? '#2563eb';
  };

  // X-axis labels: for IE, show z_alpha value; for KG, use short label
  const shortLabel = (p, i) => {
    if (isIE) return p.param.toFixed(1);
    // For KG, use a compact 2-char code
    const codes = ['ana', 'MC', 'ind', 'oCor', 'oInd'];
    return codes[i] ?? `p${i}`;
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <h1>Batch benchmark — {family} family</h1>
        <div className="session-meta">
          {policies.length} policies · {sims_per_policy} sims × {budget} steps each ·
          seed {session_seed}
          <button className="btn btn-ghost" onClick={onReset}>← New session</button>
        </div>
      </div>

      {/* Ground truth callout */}
      <div className="card" style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'baseline' }}>
          <div>
            <span style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              True best C*
            </span>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'Georgia,serif', color: '#3d2914' }}>
              {true_best_c_star.toFixed(3)}
            </div>
          </div>
          <div>
            <span style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              True min cost
            </span>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'Georgia,serif', color: '#3d2914' }}>
              {fmt(true_min_cost)}
            </div>
          </div>
          <div style={{ fontSize: 12, color: '#64748b', maxWidth: 460 }}>
            Ground truth from 30-point × 12-rep Monte Carlo. Bars below show the
            mean cost incurred by each policy's <em>final</em> best-C* pick,
            averaged over {sims_per_policy} sims. Error bars are ±1 standard deviation.
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="card">
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 8 }}>
          Mean terminal cost by policy (lower is better)
        </div>
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
          {/* Gridlines */}
          {yTicks.map((v, i) => (
            <line key={i} x1={PAD.left} x2={W - PAD.right}
                  y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
          ))}

          {/* True-min horizontal dashed line */}
          <line x1={PAD.left} x2={W - PAD.right}
                y1={yS(true_min_cost)} y2={yS(true_min_cost)}
                stroke="#dc2626" strokeWidth={1.5} strokeDasharray="6,4" opacity={0.75} />
          <text x={W - PAD.right - 4} y={yS(true_min_cost) - 5}
                textAnchor="end" fontSize={10} fill="#dc2626" fontWeight={600}>
            true min = {fmt(true_min_cost)}
          </text>

          {/* Bars + error bars */}
          {policies.map((p, i) => {
            const x = barX(i);
            const yTop = yS(p.mean_terminal_cost);
            const y0 = yS(yLo);
            const h = Math.max(1, y0 - yTop);
            const eTop = yS(p.mean_terminal_cost + p.std_terminal_cost);
            const eBot = yS(p.mean_terminal_cost - p.std_terminal_cost);
            const cx = x + barW / 2;
            return (
              <g key={i}>
                <rect x={x} y={yTop} width={barW} height={h}
                      fill={barColor(i, p)} fillOpacity={0.85} />
                {/* Error bar */}
                <line x1={cx} x2={cx} y1={eTop} y2={eBot} stroke="#374151" strokeWidth={1.5} />
                <line x1={cx - 4} x2={cx + 4} y1={eTop} y2={eTop} stroke="#374151" strokeWidth={1.5} />
                <line x1={cx - 4} x2={cx + 4} y1={eBot} y2={eBot} stroke="#374151" strokeWidth={1.5} />
              </g>
            );
          })}

          {/* X axis with labels rotated slightly for KG (long) */}
          <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom}
                stroke="#94a3b8" />
          {policies.map((p, i) => {
            const cx = barX(i) + barW / 2;
            return (
              <text key={i} x={cx} y={H - PAD.bottom + 14}
                    textAnchor="middle" fontSize={10} fill="#64748b">
                {shortLabel(p, i)}
              </text>
            );
          })}
          <text x={PAD.left + IW / 2} y={H - 8}
                textAnchor="middle" fontSize={12} fill="#64748b">
            {isIE ? 'IE parameter z_alpha' : 'KG variant'}
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
          <text transform={`translate(${PAD.left - 60},${PAD.top + IH / 2}) rotate(-90)`}
                textAnchor="middle" fontSize={12} fill="#64748b">
            Terminal cost at final best C*
          </text>
        </svg>
      </div>

      {/* Table */}
      <div className="card">
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 12 }}>
          Per-policy aggregates
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b', textAlign: 'right' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px' }}>Policy</th>
                <th style={{ padding: '8px 12px' }}>Mean best C*</th>
                <th style={{ padding: '8px 12px' }}>Mean terminal cost</th>
                <th style={{ padding: '8px 12px' }}>Std terminal cost</th>
                <th style={{ padding: '8px 12px' }}>Mean cumulative cost</th>
                <th style={{ padding: '8px 12px' }}>Std cumulative cost</th>
              </tr>
            </thead>
            <tbody>
              {policies.map((p, i) => {
                // Best (lowest) terminal cost gets bold styling
                const isMin = p.mean_terminal_cost === Math.min(...policies.map(q => q.mean_terminal_cost));
                return (
                  <tr key={i} style={{
                    borderBottom: '1px solid #f1f5f9',
                    fontWeight: isMin ? 700 : 400,
                    color: isMin ? '#16a34a' : '#374151',
                    fontFamily: 'system-ui',
                  }}>
                    <td style={{ textAlign: 'left', padding: '6px 12px' }}>{p.policy}</td>
                    <td style={{ textAlign: 'right', padding: '6px 12px', fontFamily: 'monospace' }}>
                      {p.mean_best_c_star.toFixed(3)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '6px 12px', fontFamily: 'monospace' }}>
                      {fmt(p.mean_terminal_cost)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '6px 12px', fontFamily: 'monospace' }}>
                      {fmt(p.std_terminal_cost)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '6px 12px', fontFamily: 'monospace' }}>
                      {fmt(p.mean_cumulative_cost)}
                    </td>
                    <td style={{ textAlign: 'right', padding: '6px 12px', fontFamily: 'monospace' }}>
                      {fmt(p.std_cumulative_cost)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
