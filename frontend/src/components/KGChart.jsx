// Compare five KG-related quantities as functions of θ, on one plot with
// dual y-axes (offline KG lives on cents-to-dollars scale, online KG lives
// on the μ_n scale of thousands of dollars).
//
// Left axis  (offline value-of-information):
//   Correlated (analytic)  — solid green
//   Correlated (MC)        — dashed violet
//   Independent            — solid amber
//
// Right axis (online KG, Ryzhov: mu_n(x) - (N-n) * offline_KG(x)):
//   Online (correlated)    — thin green
//   Online (independent)   — thin amber
//
// argmin of the online curves is the online-KG policy's next-measurement pick.

const W = 720, H = 370;
// PAD.top reserves room for the two-row legend that sits above the plot area.
const PAD = { top: 60, right: 96, bottom: 56, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const X_MIN = 0.01, X_MAX = 0.20;

const OFFLINE_SERIES = [
  { key: 'analytic_correlated', label: 'Offline: correlated (analytic)', color: '#16a34a', dash: null,   width: 2.2 },
  { key: 'mc_correlated',       label: 'Offline: correlated (MC)',       color: '#7c3aed', dash: '5,3',  width: 1.8 },
  { key: 'independent',         label: 'Offline: independent',           color: '#f59e0b', dash: null,   width: 2.2 },
];

const ONLINE_SERIES = [
  { key: 'online_correlated',  label: 'Online: correlated',  color: '#14532d', dash: '2,3', width: 1.6 },
  { key: 'online_independent', label: 'Online: independent', color: '#78350f', dash: '2,3', width: 1.6 },
];

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) >= 10)   return `$${v.toFixed(0)}`;
  return `$${v.toFixed(2)}`;
}

// Nice-looking axis bounds with a bit of padding, and clamp to include 0
// on the offline axis (KG is nonneg in theory).
function bounds(vals, includeZero) {
  const rawMax = Math.max(...vals);
  const rawMin = includeZero ? Math.min(0, ...vals) : Math.min(...vals);
  const pad = (rawMax - rawMin) * 0.10 || 1;
  return [rawMin - pad, rawMax + pad];
}

// argmax / argmin over the (impparams, values) samples. Skips non-finite values.
// Returns the θ (not the index) at the extremum, or null if all values are bad.
function argExtremum(impparams, values, mode /* 'max' | 'min' */) {
  if (!impparams?.length || !values?.length) return null;
  let bestVal = mode === 'max' ? -Infinity : Infinity;
  let bestI = -1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if ((mode === 'max' && v > bestVal) || (mode === 'min' && v < bestVal)) {
      bestVal = v; bestI = i;
    }
  }
  return bestI < 0 ? null : impparams[bestI];
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

  const { impparams, mc_samples, budget, steps_used } = kg;
  const remaining = Math.max(0, budget - steps_used);

  // "Next θ" per policy — the app is now a MAXIMISATION problem so every
  // curve is picked by argmax:
  //   * offline KG curves: argmax (KG is info value; more is better)
  //   * online KG curves: backend returns them in reward frame,
  //     display_OKG(x) = μ_reward(x) + (N−n)·offline_KG(x), which the
  //     policy also maximises. Same argmax convention everywhere.
  const nextByKey = {
    analytic_correlated: argExtremum(impparams, kg.analytic_correlated, 'max'),
    mc_correlated:       argExtremum(impparams, kg.mc_correlated,       'max'),
    independent:         argExtremum(impparams, kg.independent,         'max'),
    online_correlated:   argExtremum(impparams, kg.online_correlated,   'max'),
    online_independent:  argExtremum(impparams, kg.online_independent,  'max'),
  };
  // The primary KG policy used by session.policy='kg' is offline analytic
  // correlated — its argmax is what a "next step" click would sample.
  const primaryNext = nextByKey.analytic_correlated;

  // Offline axis (left)
  const offVals = OFFLINE_SERIES.flatMap(s => kg[s.key]);
  const [offLo, offHi] = bounds(offVals, /*includeZero*/ true);
  // Online axis (right)
  const onVals = ONLINE_SERIES.flatMap(s => kg[s.key]);
  const [onLo, onHi] = bounds(onVals, /*includeZero*/ false);

  const xS   = (c) => PAD.left + ((c - X_MIN) / (X_MAX - X_MIN)) * IW;
  const yOff = (v) => PAD.top + ((offHi - v) / (offHi - offLo)) * IH;
  const yOn  = (v) => PAD.top + ((onHi  - v) / (onHi  - onLo )) * IH;

  const y0Off = yOff(0);
  const offTicks = Array.from({ length: 5 }, (_, i) => offLo + (i / 4) * (offHi - offLo));
  const onTicks  = Array.from({ length: 5 }, (_, i) => onLo  + (i / 4) * (onHi  - onLo));
  const xTicks = [0.01, 0.05, 0.10, 0.15, 0.20];

  const linePath = (values, mapY) => impparams
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${xS(c)},${mapY(values[i])}`)
    .join('');

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Horizontal gridlines from the LEFT axis */}
      {offTicks.map((v, i) => (
        <line key={i} x1={PAD.left} x2={W - PAD.right}
              y1={yOff(v)} y2={yOff(v)} stroke="#e2e8f0" strokeWidth={1} />
      ))}

      {/* Zero baseline for offline KG (bold) */}
      <line x1={PAD.left} x2={W - PAD.right} y1={y0Off} y2={y0Off}
            stroke="#94a3b8" strokeWidth={1} />

      {/* "Next θ" vertical for the primary offline KG policy (analytic correlated). */}
      {primaryNext != null && (
        <>
          <line x1={xS(primaryNext)} x2={xS(primaryNext)}
                y1={PAD.top} y2={H - PAD.bottom}
                stroke="#16a34a" strokeWidth={1.5}
                strokeDasharray="6,4" opacity={0.8} />
          <text x={xS(primaryNext) + 5} y={PAD.top + 14}
                fill="#16a34a" fontSize={11} fontWeight={600}>
            next θ={primaryNext.toFixed(3)}
          </text>
        </>
      )}

      {/* Offline series on left axis */}
      {OFFLINE_SERIES.map(({ key, color, dash, width }) => (
        <g key={key}>
          <path d={linePath(kg[key], yOff)} fill="none"
                stroke={color} strokeWidth={width}
                strokeDasharray={dash ?? undefined}
                opacity={0.9} />
          {impparams.map((c, i) => (
            <circle key={i} cx={xS(c)} cy={yOff(kg[key][i])} r={2.3}
                    fill={color} opacity={0.9} />
          ))}
          {/* argmax marker for this series */}
          {nextByKey[key] != null && (() => {
            const iStar = impparams.indexOf(nextByKey[key]);
            return iStar >= 0 ? (
              <circle cx={xS(impparams[iStar])} cy={yOff(kg[key][iStar])}
                      r={5} fill="none" stroke={color} strokeWidth={2} />
            ) : null;
          })()}
        </g>
      ))}

      {/* Online series on right axis */}
      {ONLINE_SERIES.map(({ key, color, dash, width }) => (
        <g key={key}>
          <path d={linePath(kg[key], yOn)} fill="none"
                stroke={color} strokeWidth={width}
                strokeDasharray={dash ?? undefined}
                opacity={0.9} />
          {/* argmin marker for this series */}
          {nextByKey[key] != null && (() => {
            const iStar = impparams.indexOf(nextByKey[key]);
            return iStar >= 0 ? (
              <circle cx={xS(impparams[iStar])} cy={yOn(kg[key][iStar])}
                      r={5} fill="none" stroke={color} strokeWidth={2} />
            ) : null;
          })()}
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
        θ (cash buffer ratio)
      </text>

      {/* Left Y axis: offline KG */}
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {offTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 5} x2={PAD.left} y1={yOff(v)} y2={yOff(v)} stroke="#94a3b8" />
          <text x={PAD.left - 8} y={yOff(v) + 4}
                textAnchor="end" fontSize={10} fill="#64748b">
            {fmt(v)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(${PAD.left - 56},${PAD.top + IH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={12} fill="#64748b">
        Offline KG(x)
      </text>

      {/* Right Y axis: online KG */}
      <line x1={W - PAD.right} x2={W - PAD.right} y1={PAD.top} y2={H - PAD.bottom}
            stroke="#94a3b8" />
      {onTicks.map((v, i) => (
        <g key={i}>
          <line x1={W - PAD.right} x2={W - PAD.right + 5} y1={yOn(v)} y2={yOn(v)} stroke="#94a3b8" />
          <text x={W - PAD.right + 8} y={yOn(v) + 4}
                textAnchor="start" fontSize={10} fill="#64748b">
            {fmt(v)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(${W - PAD.right + 56},${PAD.top + IH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={12} fill="#64748b">
        Online KG(x) = μ_reward(x) + (N−n)·offline KG
      </text>

      {/* Legend — two horizontal rows ABOVE the plot area so it never
          overlaps the curves. Row 1 = offline series, row 2 = online
          series + N info. Columns are fixed so labels line up. */}
      <g transform={`translate(${PAD.left},8)`}>
        {OFFLINE_SERIES.map(({ label, color, dash }, i) => (
          <g key={i} transform={`translate(${i * 200},0)`}>
            <line x1={0} x2={22} y1={6} y2={6} stroke={color} strokeWidth={2}
                  strokeDasharray={dash ?? undefined} />
            <text x={28} y={9} fontSize={10} fill="#374151">
              {label}{label.includes('MC') ? `, n=${mc_samples}` : ''}
            </text>
          </g>
        ))}
        <g transform={`translate(0,20)`}>
          {ONLINE_SERIES.map(({ label, color, dash }, i) => (
            <g key={`on-${i}`} transform={`translate(${i * 200},0)`}>
              <line x1={0} x2={22} y1={6} y2={6} stroke={color} strokeWidth={1.6}
                    strokeDasharray={dash ?? undefined} />
              <text x={28} y={9} fontSize={10} fill="#374151">{label}</text>
            </g>
          ))}
          <text x={400} y={9} fontSize={10} fill="#64748b" fontStyle="italic">
            N={budget}, n={steps_used}, N−n={remaining}
          </text>
        </g>
      </g>
    </svg>
  );
}
