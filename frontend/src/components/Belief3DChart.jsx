// 3-D GP posterior surface for the 2-parameter cash-balance app.
// Uses an oblique projection to draw a wireframe/filled mesh in plain SVG.
//
// Props:
//   posterior : {
//     axis1: [number],           // θ_ind grid (length G)
//     axis2: [number],           // θ_inst grid (length G)
//     mean : [number],           // row-major (G*G), matches meshgrid(axis1, axis2, indexing='ij')
//     std  : [number],           // (same shape as mean; not used for the surface but kept in the payload)
//     history: [[θ1, θ2, cost], ...],
//     best_impparam: [θ1, θ2],
//   }

const W = 720, H = 480;
const PAD = { top: 40, right: 40, bottom: 60, left: 60 };
const IW = W - PAD.left - PAD.right;
const IH = H - PAD.top - PAD.bottom;

// Oblique / cabinet projection: draw a 3-D point (x, y, z) into 2-D screen
// coords, with x = θ_ind axis (horizontal), y = θ_inst axis (into the page),
// z = posterior mean (up).
//
//     screen_x = origin_x + sx * x + tilt_x * y
//     screen_y = origin_y - sz * z + tilt_y * y   (flip z so higher goes up)
//
// TILT_Y is deliberately small: a flatter perspective where the "into the
// page" axis doesn't rise dramatically off the base plane.
const TILT_X = 0.55;   // fraction of "y" axis pushed to the right
const TILT_Y = 0.20;   // fraction of "y" axis pushed up (kept small — flat base)

// Generic linear-interpolation colormap runner.
function interp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    if (t <= b) {
      const u = (t - a) / (b - a);
      const r  = Math.round(ca[0] + u * (cb[0] - ca[0]));
      const g  = Math.round(ca[1] + u * (cb[1] - ca[1]));
      const bl = Math.round(ca[2] + u * (cb[2] - ca[2]));
      return `rgb(${r},${g},${bl})`;
    }
  }
  return `rgb(${stops.at(-1)[1].join(',')})`;
}

const VIRIDIS_STOPS = [
  [0.00, [ 68,   1,  84]],
  [0.25, [ 59,  82, 139]],
  [0.50, [ 33, 145, 140]],
  [0.75, [ 94, 201,  98]],
  [1.00, [253, 231,  37]],
];
// Matlab-style jet: blue → cyan → green → yellow → red.
const JET_STOPS = [
  [0.00, [  0,   0, 143]],
  [0.20, [  0,  63, 255]],
  [0.40, [  0, 235, 255]],
  [0.60, [174, 255,   0]],
  [0.80, [255, 127,   0]],
  [1.00, [128,   0,   0]],
];

const COLORMAPS = {
  viridis: (t) => interp(t, VIRIDIS_STOPS),
  jet:     (t) => interp(t, JET_STOPS),
};

function fmtDollars(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}
function fmtPlain(v) {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(0);
}

/**
 * Belief3DChart — generic 3-D surface renderer over a 2-parameter box.
 *
 * @param data {
 *   axis1, axis2   : coordinate axes (length G)
 *   value          : flattened row-major G*G surface values
 *   history        : observations as [[θ1, θ2, cost], ...]  (optional)
 *   best_impparam  : [θ1, θ2] marker on the base plane  (optional)
 * }
 * @param valueLabel   : Z-axis label (e.g. "Posterior mean cost", "KG(θ)")
 * @param colorScheme  : "viridis" (default) or "jet"
 * @param obsMode      : "atCost"    — draw each observation dot at its (θ, cost) z-value on the surface,
 *                       "baseplane" — draw markers on the base plane only (best for KG surface)
 * @param dollarZ      : format Z-axis ticks as dollars (default true)
 * @param emptyMessage : text shown when data is missing
 */
export default function Belief3DChart({
  data, posterior,
  valueLabel = "Posterior mean cost",
  colorScheme = "viridis",
  obsMode = "atCost",
  dollarZ = true,
  bestLabel = "best θ",
  emptyMessage = "Waiting for surface data…",
}) {
  // Back-compat: accept the old `posterior` prop shape with `mean` field.
  const d = data ?? (posterior
    ? { ...posterior, value: posterior.mean }
    : null);

  if (!d || !d.axis1 || !d.axis2 || !d.value) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#94a3b8', fontSize: 14 }}>
        {emptyMessage}
      </div>
    );
  }

  const { axis1, axis2, value, history, best_impparam } = d;
  const G = axis1.length;
  const fmt = dollarZ ? fmtDollars : fmtPlain;
  const cmap = COLORMAPS[colorScheme] ?? COLORMAPS.viridis;

  // Value bounds (include observation z-values only for the atCost mode).
  const allZ = [...value];
  if (history && obsMode === "atCost") history.forEach(h => allZ.push(h[2]));
  const zMin = Math.min(...allZ, 0);
  const zMax = Math.max(...allZ, 1);
  const zSpan = Math.max(zMax - zMin, 1);

  // The plot area maps:  x ∈ [x1_min, x1_max] × y ∈ [x2_min, x2_max] × z ∈ [zMin, zMax]
  // onto the 2-D screen box (IW × IH after PAD).
  const x1_min = axis1[0], x1_max = axis1[axis1.length - 1];
  const x2_min = axis2[0], x2_max = axis2[axis2.length - 1];

  // Choose scales so the full projected box fits inside the plot area,
  // reserving room for the y-axis tilt.
  const sx = IW / ((x1_max - x1_min) + TILT_X * (x2_max - x2_min));
  const sz = (IH - TILT_Y * ((x2_max - x2_min) * sx)) / zSpan;

  // Origin (θ1=x1_min, θ2=x2_min, z=zMin) — bottom-left of the plot area.
  const originX = PAD.left;
  const originY = H - PAD.bottom;

  // Project (θ1, θ2, z) into (screen_x, screen_y).
  const proj = (t1, t2, z) => {
    const dx = t1 - x1_min;
    const dy = t2 - x2_min;
    const dz = z - zMin;
    return [
      originX + sx * dx + sx * TILT_X * dy,
      originY - sz * dz - sx * TILT_Y * dy,
    ];
  };

  // Build the mesh polygons.  Iterate over quad cells and draw each as a
  // filled polygon.  Painter's algorithm: draw back-to-front (large-y first).
  const polys = [];
  const at = (i, j) => value[i * G + j];  // row-major, i indexes axis1, j indexes axis2
  for (let i = 0; i < G - 1; i++) {
    for (let j = G - 2; j >= 0; j--) {  // reverse-j so back rows draw first
      const z00 = at(i, j),     z01 = at(i, j + 1);
      const z10 = at(i + 1, j), z11 = at(i + 1, j + 1);
      const p00 = proj(axis1[i],     axis2[j],     z00);
      const p01 = proj(axis1[i],     axis2[j + 1], z01);
      const p11 = proj(axis1[i + 1], axis2[j + 1], z11);
      const p10 = proj(axis1[i + 1], axis2[j],     z10);
      const zAvg = (z00 + z01 + z10 + z11) / 4;
      const t = (zAvg - zMin) / zSpan;
      const points = `${p00[0]},${p00[1]} ${p10[0]},${p10[1]} ${p11[0]},${p11[1]} ${p01[0]},${p01[1]}`;
      polys.push(
        <polygon key={`${i}-${j}`} points={points}
                 fill={cmap(t)} stroke="rgba(255,255,255,0.35)"
                 strokeWidth={0.5} />
      );
    }
  }

  // Frame edges of the box (draw dashed reference lines for the 3-D box).
  const cornerX = proj(x1_max, x2_min, zMin);
  const cornerY = proj(x1_min, x2_max, zMin);
  const cornerXY = proj(x1_max, x2_max, zMin);
  const originProj = proj(x1_min, x2_min, zMin);

  // Axis tick labels
  const NUM_TICKS = 5;
  const x1_ticks = Array.from({ length: NUM_TICKS }, (_, k) =>
    x1_min + (k / (NUM_TICKS - 1)) * (x1_max - x1_min));
  const x2_ticks = Array.from({ length: NUM_TICKS }, (_, k) =>
    x2_min + (k / (NUM_TICKS - 1)) * (x2_max - x2_min));
  const z_ticks = Array.from({ length: 5 }, (_, k) =>
    zMin + (k / 4) * zSpan);

  // Observation dots and best-θ marker
  const bestProj = best_impparam
    ? proj(best_impparam[0], best_impparam[1], zMin)
    : null;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      {/* Base rectangle (θ1, θ2 plane at z=zMin) */}
      <polygon
        points={`${originProj[0]},${originProj[1]} ${cornerX[0]},${cornerX[1]} ${cornerXY[0]},${cornerXY[1]} ${cornerY[0]},${cornerY[1]}`}
        fill="none" stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,3" opacity={0.6}
      />

      {/* Mesh surface */}
      {polys}

      {/* Z axis (vertical, at back-left corner) */}
      {(() => {
        const bottom = proj(x1_min, x2_min, zMin);
        const top    = proj(x1_min, x2_min, zMax);
        return (
          <>
            <line x1={bottom[0]} y1={bottom[1]} x2={top[0]} y2={top[1]}
                  stroke="#334155" strokeWidth={1.2} />
            {z_ticks.map((zv, k) => {
              const pt = proj(x1_min, x2_min, zv);
              return (
                <g key={k}>
                  <line x1={pt[0] - 4} y1={pt[1]} x2={pt[0]} y2={pt[1]}
                        stroke="#334155" strokeWidth={1} />
                  <text x={pt[0] - 6} y={pt[1] + 4}
                        textAnchor="end" fontSize={10} fill="#64748b">
                    {fmt(zv)}
                  </text>
                </g>
              );
            })}
            <text x={bottom[0] - 34} y={(bottom[1] + top[1]) / 2}
                  transform={`rotate(-90, ${bottom[0] - 34}, ${(bottom[1] + top[1]) / 2})`}
                  textAnchor="middle" fontSize={12} fill="#64748b">
              {valueLabel}
            </text>
          </>
        );
      })()}

      {/* θ_ind axis (front-bottom) */}
      {x1_ticks.map((v, k) => {
        const pt = proj(v, x2_min, zMin);
        return (
          <g key={`x1-${k}`}>
            <line x1={pt[0]} y1={pt[1]} x2={pt[0]} y2={pt[1] + 5} stroke="#334155" strokeWidth={1} />
            <text x={pt[0]} y={pt[1] + 16} textAnchor="middle" fontSize={10} fill="#64748b">
              {v.toFixed(2)}
            </text>
          </g>
        );
      })}
      <text x={(originProj[0] + cornerX[0]) / 2} y={originProj[1] + 34}
            textAnchor="middle" fontSize={12} fill="#64748b">
        θ₁ (individual buffer)
      </text>

      {/* θ_inst axis (right-back, projected diagonally) */}
      {x2_ticks.map((v, k) => {
        const pt = proj(x1_max, v, zMin);
        return (
          <g key={`x2-${k}`}>
            <line x1={pt[0]} y1={pt[1]} x2={pt[0] + 5} y2={pt[1] + 3} stroke="#334155" strokeWidth={1} />
            <text x={pt[0] + 10} y={pt[1] + 6} textAnchor="start" fontSize={10} fill="#64748b">
              {v.toFixed(2)}
            </text>
          </g>
        );
      })}
      <text x={(cornerX[0] + cornerXY[0]) / 2 + 20} y={(cornerX[1] + cornerXY[1]) / 2 + 6}
            textAnchor="start" fontSize={12} fill="#64748b">
        θ₂ (institutional buffer)
      </text>

      {/* Observations — atCost (dot at the observed z value, w/ drop line) or
          baseplane (marker on the base plane only, for KG surfaces where the
          observation has no natural z). */}
      {(history ?? []).map((h, k) => {
        const [t1, t2, cost] = h;
        const pBase = proj(t1, t2, zMin);
        if (obsMode === "baseplane") {
          return (
            <g key={`obs-${k}`}>
              <circle cx={pBase[0]} cy={pBase[1]} r={4}
                      fill="#dc2626" stroke="white" strokeWidth={1.2} opacity={0.9} />
            </g>
          );
        }
        const p = proj(t1, t2, cost);
        return (
          <g key={`obs-${k}`}>
            <line x1={pBase[0]} y1={pBase[1]} x2={p[0]} y2={p[1]}
                  stroke="#dc2626" strokeWidth={1} strokeDasharray="3,2" opacity={0.55} />
            <circle cx={p[0]} cy={p[1]} r={5}
                    fill="#dc2626" stroke="white" strokeWidth={1.4} />
          </g>
        );
      })}

      {/* Marker on the base plane — semantics chosen by caller via bestLabel. */}
      {bestProj && (
        <g>
          <circle cx={bestProj[0]} cy={bestProj[1]} r={6}
                  fill="none" stroke="#16a34a" strokeWidth={2} />
          <text x={bestProj[0] + 8} y={bestProj[1] - 6}
                fontSize={11} fill="#16a34a" fontWeight={600}>
            {bestLabel}
          </text>
        </g>
      )}

      {/* Legend */}
      <g transform={`translate(${W - PAD.right - 150}, 10)`}>
        <rect x={0} y={0} width={12} height={12} fill={cmap(0.5)} opacity={0.85} />
        <text x={16} y={10} fontSize={10} fill="#374151">{valueLabel}</text>
        <circle cx={6} cy={26} r={4} fill="#dc2626" />
        <text x={16} y={30} fontSize={10} fill="#374151">Observation ({(history ?? []).length})</text>
        <circle cx={6} cy={42} r={5} fill="none" stroke="#16a34a" strokeWidth={2} />
        <text x={16} y={46} fontSize={10} fill="#374151">{bestLabel}</text>
      </g>
    </svg>
  );
}
