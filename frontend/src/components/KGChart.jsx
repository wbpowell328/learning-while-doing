// Five decision-value curves side by side, so the user can see how the
// candidate policies pick their next θ:
//
// Left axis  — offline value of information (per-batch dollars):
//   analytic_correlated   — solid green — offline KG(x; ρˡᵏʰᵈ)
//
// Right axis — reward-frame per-batch dollars (μ plus/minus a policy term):
//   online_correlated     — solid dark green   — μ + KG              (Warren-2026)
//   ryzhov                — solid orange       — μ + (N−n)·KG        (Ryzhov 2010)
//   ie_0                  — dashed blue        — μ                   (IE, ρ^IE = 0)
//   ie_1_5                — dashed navy        — μ + 1.5·σ           (IE, ρ^IE = 1.5)
//
// Every curve is maximised at its next-θ choice, so argmax is uniform.

const W = 720, H = 370;
// PAD.top reserves room for the two-row legend that sits above the plot area.
const PAD = { top: 60, right: 96, bottom: 56, left: 72 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

const X_MIN = 0.01, X_MAX = 0.20;

const OFFLINE_SERIES = [
  { key: 'analytic_correlated', label: 'Offline: KG(x)', color: '#16a34a', dash: null, width: 2.2 },
];

const ONLINE_SERIES = [
  { key: 'online_correlated', label: 'Online: μ + KG',                  color: '#14532d', dash: null,   width: 1.8 },
  { key: 'ryzhov',            label: 'Online: μ + (N−n)·KG (Ryzhov)',   color: '#c2410c', dash: null,   width: 1.8 },
  { key: 'ie_0',              label: 'Greedy',                          color: '#2563eb', dash: '4,3',  width: 1.6 },
  { key: 'ie_1_5',            label: 'IE (ρ^IE = 1.5) = μ + 1.5·σ',     color: '#1e3a8a', dash: '4,3',  width: 1.6 },
];

// Explicit legend layout — decoupled from the OFFLINE/ONLINE split so
// we can group five items into two visually balanced rows without
// changing how the curves themselves are computed.
const _SERIES_BY_KEY = Object.fromEntries(
  [...OFFLINE_SERIES, ...ONLINE_SERIES].map(s => [s.key, s])
);
const LEGEND_ROWS = [
  ['ryzhov', 'online_correlated', 'analytic_correlated'],
  ['ie_1_5', 'ie_0'],
].map(row => row.map(k => _SERIES_BY_KEY[k]));

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

export default function KGChart({ kg, nextTheta, lastTheta }) {
  if (!kg) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 14 }}>
        Create a session to see KG values.
      </div>
    );
  }

  const { impparams } = kg;

  // "Next θ" per policy — everything's a maximisation, so argmax across
  // every curve. These drive the small argmax rings on each curve.
  const nextByKey = Object.fromEntries(
    [...OFFLINE_SERIES, ...ONLINE_SERIES].map(s =>
      [s.key, argExtremum(impparams, kg[s.key], 'max')]
    )
  );
  // The prominent green "next θ" vertical marks where the ACTIVE policy
  // will actually sample next — the backend's authoritative next_theta
  // preview (deterministic for the KG family and IE). This is keyed to
  // whatever policy the user is running, NOT hard-wired to offline KG,
  // so on (say) Ryzhov the line lands on the Ryzhov pick instead of the
  // offline-KG exploration point. Falls back to nothing (e.g. Manual /
  // Human, which propose no θ).
  const _nt = Array.isArray(nextTheta) ? nextTheta[0] : nextTheta;
  const primaryNext = (typeof _nt === 'number' && Number.isFinite(_nt)) ? _nt : null;

  // The θ the LAST experiment actually sampled. Drawn as a separate
  // slate marker so the user can see past-vs-future at a glance: the
  // chart already folds in this observation, so after sampling here the
  // acquisition value drops and the argmax (next θ) generally moves —
  // that's why "next θ" sits away from "last θ" on exploration steps.
  const _lt = Array.isArray(lastTheta) ? lastTheta[0] : lastTheta;
  const lastSampled = (typeof _lt === 'number' && Number.isFinite(_lt)) ? _lt : null;

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

      {/* "Last θ" — where the most recent experiment actually sampled.
          Slate, label pinned to the bottom so it never collides with the
          green "next θ" label at the top. Seeing both makes the temporal
          story explicit: we sampled HERE, the belief updated, so the
          policy's NEXT pick (green) is generally somewhere else. */}
      {lastSampled != null && (
        <>
          <line x1={xS(lastSampled)} x2={xS(lastSampled)}
                y1={PAD.top} y2={H - PAD.bottom}
                stroke="#64748b" strokeWidth={1.5}
                strokeDasharray="2,3" opacity={0.7} />
          <text x={xS(lastSampled) + 5} y={H - PAD.bottom - 6}
                fill="#64748b" fontSize={11} fontWeight={600}>
            last θ={lastSampled.toFixed(3)}
          </text>
        </>
      )}

      {/* "Next θ" vertical — the ACTIVE policy's next pick (backend
          next_theta preview). */}
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
          {/* Per-point dots only when the grid is coarse; on the fine
              policy grid (~100 pts) the line alone reads cleaner. */}
          {impparams.length <= 40 && impparams.map((c, i) => (
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
        Policy value per batch (μ ± term)
      </text>

      {/* Legend — five entries in two rows above the plot area,
          grouped by LEGEND_ROWS: row 1 has the three "big" curves
          (Ryzhov, Warren, offline KG); row 2 has IE and Greedy. Column
          width scales with row length so labels don't crowd. */}
      <g transform={`translate(${PAD.left},8)`}>
        {LEGEND_ROWS.map((row, ri) => {
          const col = Math.max(140, Math.floor(IW / row.length));
          return (
            <g key={`row-${ri}`} transform={`translate(0,${ri * 20})`}>
              {row.map(({ key, label, color, dash, width }, ci) => (
                <g key={key} transform={`translate(${ci * col},0)`}>
                  <line x1={0} x2={22} y1={6} y2={6} stroke={color}
                        strokeWidth={width ?? 1.8}
                        strokeDasharray={dash ?? undefined} />
                  <text x={28} y={9} fontSize={10} fill="#374151">{label}</text>
                </g>
              ))}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
