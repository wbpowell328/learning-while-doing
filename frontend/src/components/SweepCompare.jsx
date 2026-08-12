import { useState, useMemo } from 'react';

// Compare-sweeps panel — a chart view over the Run-history log. Every
// seed sweep already persists its per-seed rows plus a Sweep-avg row,
// each stamped with a shared sweep_id, so we can group them and draw a
// clustered bar chart of profit: one cluster per sweep, one bar per
// seed, plus the average. No re-simulation — this is purely a view.

// Stable seed colours (light theme, matching the app). Seeds map to
// colours by their position in the sorted union of seeds across the
// selected sweeps, so seed 24 is the same colour in every cluster.
const SEED_COLORS = [
  '#2a78d6', '#eb6834', '#1baf7a', '#eda100',
  '#e87ba4', '#008300', '#4a3aa7', '#e34948',
];
const AVG_COLOR = '#0f172a';   // dark slate — reads as the summary bar

const POLICY_SHORT = {
  okg_ryzhov: 'Ryzhov', okg: 'Online KG', okg_indep: 'Online KG (indep)',
  kg: 'Offline KG', kg_indep: 'Offline KG (indep)', ie: 'IE', ie_15: 'IE (1.5)',
  greedy: 'Greedy', randomized_greedy: 'Rand. greedy', random: 'Random', human: 'Manual',
};

// Parameters the user can put on the x-axis. label is what the dropdown
// and axis show; key is the run-history field.
const X_PARAMS = [
  { key: 'horizon', label: 'Horizon' },
  { key: 'repeat',  label: 'Repeat' },
  { key: 'budget',  label: 'N (Ryzhov)' },
  { key: 'rho_ell', label: 'ρᵇᵃⁿᵈ (bandwidth)' },
  { key: 'mstar',   label: 'ρˡᵏʰᵈ (lookahead)' },
];

function fmtDollars(v) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  const sign = n < 0 ? '−' : '';
  return `${sign}$${Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}
function fmtParamVal(v) {
  if (v == null) return '—';
  if (Array.isArray(v)) return `(${v.map(x => Number(x)).join(', ')})`;
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/\.?0+$/, '');
}

// Group Run-history rows into sweeps.
//   Preferred: the stamped sweep_id (bulletproof).
//   Fallback for rows logged before sweep_id existed: contiguity. A
//   sweep's rows are logged as one unbroken block — newest first, the
//   Sweep-avg row then its seed rows — so we cut a new group at each
//   Sweep-avg row, parameter change, or non-sweep row. (The old
//   timestamp-minute fallback wrongly split any sweep that crossed a
//   clock-minute boundary, e.g. a slow 5-seed reveal spanning 19:46–48.)
function groupSweeps(rows) {
  const groups = [];
  const byId = new Map();
  let cur = null;   // active contiguity group (rows without a sweep_id)
  const sameParams = (a, b) =>
    a.policy === b.policy && a.horizon === b.horizon && a.repeat === b.repeat &&
    a.budget === b.budget && String(a.rho_ell) === String(b.rho_ell) && a.mstar === b.mstar;

  for (const r of rows) {
    if (r.action !== 'Sweep' && r.action !== 'Sweep avg') { cur = null; continue; }

    if (r.sweep_id != null) {
      let g = byId.get(r.sweep_id);
      if (!g) {
        g = { key: `id:${r.sweep_id}`, seedRows: [], avgRow: null, sample: r };
        byId.set(r.sweep_id, g); groups.push(g);
      }
      if (r.action === 'Sweep avg') g.avgRow = r; else g.seedRows.push(r);
      cur = null;
      continue;
    }

    if (r.action === 'Sweep avg') {
      cur = { key: `c:${r.ts}`, seedRows: [], avgRow: r, sample: r };
      groups.push(cur);
    } else {
      if (!cur || !sameParams(cur.sample, r)) {
        cur = { key: `c:${r.ts}`, seedRows: [], avgRow: null, sample: r };
        groups.push(cur);
      }
      cur.seedRows.push(r);
    }
  }

  // Newest sweeps first (by the sample row's timestamp).
  return groups.sort((a, b) =>
    String(b.sample.ts ?? '').localeCompare(String(a.sample.ts ?? '')));
}

function sweepLabel(g) {
  const s = g.sample;
  const pol = POLICY_SHORT[s.policy] ?? s.policy ?? '?';
  return `${pol} · H=${s.horizon ?? '?'}×${s.repeat ?? '?'}` +
    (s.budget != null ? ` · N=${s.budget}` : '');
}

const btn = (active) => ({
  padding: '3px 10px', border: `1px solid ${active ? '#334155' : '#cbd5e1'}`,
  borderRadius: 4, fontSize: 12, cursor: 'pointer',
  background: active ? '#0f172a' : '#fff', color: active ? '#fff' : '#334155',
});
const miniBtn = {
  padding: '2px 8px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 11.5, cursor: 'pointer', background: '#fff', color: '#334155',
};

// ── Clustered bar chart (inline SVG) ──────────────────────────────────
function ClusteredBars({ clusters, seeds, xLabel }) {
  // clusters: [{ x, xText, bars:[{label,color,value,isAvg}] }]
  const W = Math.max(560, clusters.length * 108);
  const H = 340;
  const m = { top: 14, right: 14, bottom: 54, left: 70 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;

  const vals = clusters.flatMap(c => c.bars.map(b => b.value)).filter(Number.isFinite);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (lo === hi) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= (lo < 0 ? pad : 0); hi += pad;
  const y = v => m.top + ph - ((v - lo) / (hi - lo)) * ph;
  const y0 = y(0);

  // ~5 gridline ticks, rounded to something readable.
  const ticks = [];
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) ticks.push(lo + (i / nTicks) * (hi - lo));

  const clusterW = pw / clusters.length;
  const innerW = clusterW * 0.82;
  const gap = 2;

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} role="img"
           style={{ maxWidth: '100%', minWidth: Math.min(W, 560) }}
           aria-label={`Clustered bar chart of profit by seed, grouped by ${xLabel}.`}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={m.left} x2={m.left + pw} y1={y(t)} y2={y(t)}
                  stroke={Math.abs(t) < 1e-9 ? '#94a3b8' : '#e2e8f0'} strokeWidth="1" />
            <text x={m.left - 8} y={y(t) + 4} textAnchor="end"
                  fontSize="11" fill="#64748b" style={{ fontVariantNumeric: 'tabular-nums' }}>
              {fmtDollars(t)}
            </text>
          </g>
        ))}
        {clusters.map((c, ci) => {
          const cx0 = m.left + ci * clusterW + (clusterW - innerW) / 2;
          const bw = (innerW - gap * (c.bars.length - 1)) / c.bars.length;
          return (
            <g key={ci}>
              {c.bars.map((b, bi) => {
                if (!Number.isFinite(b.value)) return null;
                const bx = cx0 + bi * (bw + gap);
                const top = Math.min(y(b.value), y0);
                const h = Math.abs(y(b.value) - y0);
                return (
                  <rect key={bi} x={bx} y={top} width={Math.max(1, bw)} height={Math.max(0, h)}
                        rx="1.5" fill={b.color}>
                    <title>{`${b.label} · ${xLabel}=${c.xText}: ${fmtDollars(b.value)}`}</title>
                  </rect>
                );
              })}
              <text x={m.left + ci * clusterW + clusterW / 2} y={H - m.bottom + 18}
                    textAnchor="middle" fontSize="12" fill="#334155"
                    style={{ fontVariantNumeric: 'tabular-nums' }}>
                {c.xText}
              </text>
            </g>
          );
        })}
        <text x={m.left + pw / 2} y={H - 8} textAnchor="middle" fontSize="12" fill="#475569">
          {xLabel}
        </text>
      </svg>
    </div>
  );
}

export default function SweepCompare({ rows = [] }) {
  const [open, setOpen] = useState(false);
  const sweeps = useMemo(() => groupSweeps(rows), [rows]);

  // Selection: which sweeps to plot. Default to all once opened.
  const [selected, setSelected] = useState(null);   // null = "not yet initialised"
  const sel = selected ?? new Set(sweeps.map(g => g.key));

  // X-axis parameter: default to the first param that actually varies
  // across the selected sweeps.
  const [xParam, setXParam] = useState(null);
  const selectedSweeps = sweeps.filter(g => sel.has(g.key));
  const autoX = useMemo(() => {
    for (const p of X_PARAMS) {
      const vals = new Set(selectedSweeps.map(g => JSON.stringify(g.sample[p.key] ?? null)));
      if (vals.size > 1) return p.key;
    }
    return 'horizon';
  }, [selectedSweeps]);
  const xKey = xParam ?? autoX;
  const xMeta = X_PARAMS.find(p => p.key === xKey) ?? X_PARAMS[0];

  function toggle(key) {
    const next = new Set(sel);
    next.has(key) ? next.delete(key) : next.add(key);
    setSelected(next);
  }
  // "Recent N" quick-pick — sweeps are listed newest-first, so the N most
  // recently run are the first N keys.
  const [recentN, setRecentN] = useState('7');
  function applyRecent() {
    const n = Math.max(1, Math.min(sweeps.length, Math.round(Number(recentN) || 0)));
    setSelected(new Set(sweeps.slice(0, n).map(g => g.key)));
  }

  // Build chart data from the selected sweeps.
  const allSeeds = useMemo(() => {
    const s = new Set();
    selectedSweeps.forEach(g => g.seedRows.forEach(r => {
      if (Number.isFinite(Number(r.seed))) s.add(Number(r.seed));
    }));
    return [...s].sort((a, b) => a - b);
  }, [selectedSweeps]);
  const seedColor = seed => SEED_COLORS[allSeeds.indexOf(seed) % SEED_COLORS.length];

  const clusters = useMemo(() => {
    return selectedSweeps
      .map(g => {
        const xv = g.sample[xKey];
        const bars = allSeeds.map(seed => {
          const row = g.seedRows.find(r => Number(r.seed) === seed);
          return { label: `Seed ${seed}`, color: seedColor(seed),
                   value: row ? Number(row.cumulative) : NaN };
        });
        if (g.avgRow) {
          bars.push({ label: 'Sweep avg', color: AVG_COLOR,
                      value: Number(g.avgRow.cumulative), isAvg: true });
        }
        return { key: g.key, xVal: Number(xv), xText: fmtParamVal(xv), bars };
      })
      .sort((a, b) => (Number.isFinite(a.xVal) && Number.isFinite(b.xVal))
        ? a.xVal - b.xVal
        : String(a.xText).localeCompare(String(b.xText)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSweeps, xKey, allSeeds]);

  return (
    <div className="card" style={{ padding: '10px 14px', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
          Compare sweeps
        </div>
        <span style={{ fontSize: 12, color: '#64748b' }}>
          {sweeps.length === 0
            ? '(run a seed sweep to populate this)'
            : `(${sweeps.length} ${sweeps.length === 1 ? 'sweep' : 'sweeps'} in the log; profit per seed, clustered)`}
        </span>
        <button type="button" onClick={() => setOpen(v => !v)}
                disabled={sweeps.length === 0}
                style={{ marginLeft: 'auto', ...btn(false),
                         opacity: sweeps.length === 0 ? 0.5 : 1,
                         cursor: sweeps.length === 0 ? 'not-allowed' : 'pointer' }}>
          {open ? 'Close' : 'Open'}
        </button>
      </div>

      {open && sweeps.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 10 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>
                  Sweeps to plot
                </span>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  ({sel.size} of {sweeps.length})
                </span>
                <button type="button" onClick={() => setSelected(new Set(sweeps.map(g => g.key)))}
                        style={miniBtn} title="Select every sweep">All</button>
                <button type="button" onClick={() => setSelected(new Set())}
                        style={miniBtn} title="Clear the selection">None</button>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#475569' }}
                      title="Select the N most recently run sweeps (top of the list).">
                  Recent
                  <input type="number" min={1} max={sweeps.length}
                         value={recentN}
                         onChange={e => setRecentN(e.target.value)}
                         onKeyDown={e => { if (e.key === 'Enter') applyRecent(); }}
                         style={{ width: 44, padding: '2px 4px', border: '1px solid #cbd5e1',
                                  borderRadius: 4, fontSize: 12 }} />
                  <button type="button" onClick={applyRecent} style={miniBtn}>Apply</button>
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 168,
                            overflowY: 'auto', paddingRight: 6 }}>
                {sweeps.map((g, i) => (
                  <label key={g.key} style={{ display: 'flex', alignItems: 'center', gap: 6,
                                              fontSize: 12.5, color: '#0f172a', cursor: 'pointer' }}>
                    <input type="checkbox" checked={sel.has(g.key)} onChange={() => toggle(g.key)} />
                    <span style={{ color: '#94a3b8', minWidth: 16, textAlign: 'right' }}>{i + 1}.</span>
                    {sweepLabel(g)}
                    <span style={{ color: '#94a3b8' }}>
                      ({g.seedRows.length} {g.seedRows.length === 1 ? 'seed' : 'seeds'})
                    </span>
                  </label>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>
                X-axis
              </div>
              <select value={xKey} onChange={e => setXParam(e.target.value)}
                      style={{ padding: '4px 8px', border: '1px solid #cbd5e1',
                               borderRadius: 4, fontSize: 13 }}>
                {X_PARAMS.map(p => (
                  <option key={p.key} value={p.key}>{p.label}</option>
                ))}
              </select>
              {xParam == null && (
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, maxWidth: 160 }}>
                  auto: the parameter that varies across your selection
                </div>
              )}
            </div>
          </div>

          {clusters.length === 0 ? (
            <p style={{ fontSize: 12.5, color: '#64748b' }}>
              Select at least one sweep to plot.
            </p>
          ) : (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14,
                            marginBottom: 8, fontSize: 12, color: '#475569' }}>
                {allSeeds.map(seed => (
                  <span key={seed} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 11, height: 11, borderRadius: 2,
                                   background: seedColor(seed) }} />
                    Seed {seed}
                  </span>
                ))}
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 13, height: 13, borderRadius: 2, background: AVG_COLOR }} />
                  Sweep avg
                </span>
              </div>
              <ClusteredBars clusters={clusters} seeds={allSeeds} xLabel={xMeta.label} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
