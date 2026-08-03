// Batch benchmarking results view.
//
// Panels:
//   1. Bar chart: mean terminal cost by policy (± 1σ), red dashed reference
//      at ground-truth min cost.
//   2. Bar chart: mean cumulative cost by policy (± 1σ), red dashed reference
//      at (budget × ground-truth min cost) — the theoretical floor if you had
//      known the optimum from step 1 and always played it.
//   3. Table of per-policy aggregates.

const W = 720, H = 320;
const PAD = { top: 24, right: 24, bottom: 76, left: 84 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(2)}k`;
  return `$${v.toFixed(1)}`;
}

/**
 * MetricBarChart — bar chart with error bars and an optional reference line.
 *
 * @param policies       array of BatchPolicyResult
 * @param meanField      key on each policy for the mean (e.g. "mean_terminal_cost")
 * @param stdField       key on each policy for the std (e.g. "std_terminal_cost")
 * @param yLabel         Y-axis label
 * @param referenceValue optional numeric reference to draw as dashed horizontal line
 * @param referenceLabel label placed to the right of the reference line
 * @param family         "KG" or "IE" — drives x labels and bar colors
 */
function MetricBarChart({ policies, meanField, stdField, yLabel,
                         referenceValue = null, referenceLabel = '', family }) {
  // Y bounds — include error bar extremes and (optionally) the reference line
  const yVals = policies.flatMap(p => [
    p[meanField] + p[stdField],
    p[meanField] - p[stdField],
  ]);
  if (referenceValue != null) yVals.push(referenceValue);
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

  const isIE = family === 'IE';
  const kgColorMap = [ '#16a34a', '#7c3aed', '#f59e0b', '#14532d', '#78350f' ];
  const barColor = (i) => isIE ? '#2563eb' : (kgColorMap[i] ?? '#2563eb');

  const shortLabel = (p, i) => {
    if (isIE) return p.param.toFixed(1);
    const codes = ['ana', 'MC', 'ind', 'oCor', 'oInd'];
    return codes[i] ?? `p${i}`;
  };

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Gridlines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yS(v)} y2={yS(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Reference dashed line */}
      {referenceValue != null && (
        <>
          <line x1={PAD.left} x2={W - PAD.right}
                y1={yS(referenceValue)} y2={yS(referenceValue)}
                stroke="#dc2626" strokeWidth={1.5} strokeDasharray="6,4" opacity={0.75} />
          <text x={W - PAD.right - 4} y={yS(referenceValue) - 5}
                textAnchor="end" fontSize={10} fill="#dc2626" fontWeight={600}>
            {referenceLabel}
          </text>
        </>
      )}

      {/* Bars + error bars */}
      {policies.map((p, i) => {
        const x = barX(i);
        const yTop = yS(p[meanField]);
        const y0 = yS(yLo);
        const h = Math.max(1, y0 - yTop);
        const eTop = yS(p[meanField] + p[stdField]);
        const eBot = yS(p[meanField] - p[stdField]);
        const cx = x + barW / 2;
        return (
          <g key={i}>
            <rect x={x} y={yTop} width={barW} height={h}
                  fill={barColor(i)} fillOpacity={0.85} />
            <line x1={cx} x2={cx} y1={eTop} y2={eBot} stroke="#374151" strokeWidth={1.5} />
            <line x1={cx - 4} x2={cx + 4} y1={eTop} y2={eTop} stroke="#374151" strokeWidth={1.5} />
            <line x1={cx - 4} x2={cx + 4} y1={eBot} y2={eBot} stroke="#374151" strokeWidth={1.5} />
          </g>
        );
      })}

      {/* X axis */}
      <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {policies.map((p, i) => (
        <text key={i} x={barX(i) + barW / 2} y={H - PAD.bottom + 14}
              textAnchor="middle" fontSize={10} fill="#64748b">
          {shortLabel(p, i)}
        </text>
      ))}
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
        {yLabel}
      </text>
    </svg>
  );
}


export default function BatchResults({ batch, onReset }) {
  if (!batch) return null;
  const { family, sims_per_policy, budget, session_seed,
          true_best_impparam, true_min_cost, policies } = batch;

  // Cumulative-cost lower bound: play at the true optimum every step.
  const cumulativeFloor = budget * true_min_cost;

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
              True best θ
            </span>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'Georgia,serif', color: '#3d2914' }}>
              {true_best_impparam.toFixed(3)}
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
            Ground truth from 30-point × 12-rep Monte Carlo. Charts below show
            terminal cost (quality of the <em>final</em> pick) and cumulative
            cost (total incurred during the {budget}-step run), each averaged
            over {sims_per_policy} sims. Error bars are ±1 standard deviation.
          </div>
        </div>
      </div>

      {/* Chart 1: terminal cost */}
      <div className="card">
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 8 }}>
          Mean terminal cost by policy (lower is better)
        </div>
        <MetricBarChart
          policies={policies}
          meanField="mean_terminal_cost"
          stdField="std_terminal_cost"
          yLabel="Terminal cost at final best θ"
          referenceValue={true_min_cost}
          referenceLabel={`true min = ${fmt(true_min_cost)}`}
          family={family}
        />
      </div>

      {/* Chart 2: cumulative cost */}
      <div className="card">
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 8 }}>
          Mean cumulative cost by policy (lower = cheaper learning)
        </div>
        <MetricBarChart
          policies={policies}
          meanField="mean_cumulative_cost"
          stdField="std_cumulative_cost"
          yLabel={`Cumulative cost over ${budget} steps`}
          referenceValue={cumulativeFloor}
          referenceLabel={`floor = ${fmt(cumulativeFloor)} (= ${budget} × true min)`}
          family={family}
        />
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
                <th style={{ padding: '8px 12px' }}>Mean best θ</th>
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
                      {p.mean_best_impparam.toFixed(3)}
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
