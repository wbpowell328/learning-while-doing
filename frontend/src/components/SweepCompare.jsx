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
// Sample mean / std (n−1), matching SeedSweep's summary row.
function meanStd(xs) {
  const g = xs.filter(Number.isFinite);
  if (!g.length) return { mean: NaN, std: NaN, n: 0 };
  const mean = g.reduce((a, b) => a + b, 0) / g.length;
  if (g.length < 2) return { mean, std: NaN, n: g.length };
  const v = g.reduce((s, x) => s + (x - mean) ** 2, 0) / (g.length - 1);
  return { mean, std: Math.sqrt(v), n: g.length };
}
// Individual Run / Manual rows become single-point "sweeps" so they can
// be dropped into the comparison alongside real sweeps. One seed, no
// average.
function groupRuns(rows) {
  return rows
    .filter(r => r.action === 'Run' || r.action === 'Manual')
    .map(r => ({ key: `run:${r.ts}`, seedRows: [r], avgRow: null, isRun: true, sample: r }));
}
// CSV helpers — quote a cell only when it contains a comma, quote, or
// newline (RFC 4180), and round numeric profit to cents.
function csvCell(v) {
  if (v == null) return '';
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
const round2 = v => (Number.isFinite(Number(v)) ? Math.round(Number(v) * 100) / 100 : '');
function downloadCsv(filename, text) {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
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

// ── Clustered chart (inline SVG) — bars or dots±std ───────────────────
function SweepChart({ clusters, xLabel, mode }) {
  // clusters: [{ num, xText, bars:[{label,color,value,isAvg}] }]
  const W = Math.max(560, clusters.length * 108);
  const H = 356;
  const m = { top: 14, right: 14, bottom: 70, left: 70 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;

  // Per-cluster seed values + mean/std (dots mode draws these).
  const cc = clusters.map(c => {
    const seedBars = c.bars.filter(b => !b.isAvg);
    return { ...c, seedBars, st: meanStd(seedBars.map(b => b.value)) };
  });

  const vals = [];
  cc.forEach(c => {
    c.bars.forEach(b => { if (Number.isFinite(b.value)) vals.push(b.value); });
    if (mode === 'dots' && Number.isFinite(c.st.mean)) {
      vals.push(c.st.mean);
      if (Number.isFinite(c.st.std)) vals.push(c.st.mean + c.st.std, c.st.mean - c.st.std);
    }
  });
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (lo === hi) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= (lo < 0 ? pad : 0); hi += pad;
  const y = v => m.top + ph - ((v - lo) / (hi - lo)) * ph;
  const y0 = y(0);

  const ticks = [];
  for (let i = 0; i <= 5; i++) ticks.push(lo + (i / 5) * (hi - lo));

  const clusterW = pw / cc.length;
  const innerW = clusterW * 0.82;
  const gap = 2;

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} role="img"
           style={{ maxWidth: '100%', minWidth: Math.min(W, 560) }}
           aria-label={`Profit by seed, grouped by ${xLabel}, shown as ${mode === 'dots' ? 'dots with mean and standard-deviation bars' : 'clustered bars'}.`}>
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
        {cc.map((c, ci) => {
          const centerX = m.left + ci * clusterW + clusterW / 2;
          return (
            <g key={ci}>
              {mode === 'bars' && (() => {
                const cx0 = m.left + ci * clusterW + (clusterW - innerW) / 2;
                const bw = (innerW - gap * (c.bars.length - 1)) / c.bars.length;
                return c.bars.map((b, bi) => {
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
                });
              })()}
              {mode === 'dots' && (
                <>
                  {c.seedBars.map((b, bi) => {
                    if (!Number.isFinite(b.value)) return null;
                    const jx = centerX - 10 + (bi - (c.seedBars.length - 1) / 2) * 3.5;
                    return (
                      <circle key={bi} cx={jx} cy={y(b.value)} r="4"
                              fill={b.color} fillOpacity="0.9" stroke="#fff" strokeWidth="1">
                        <title>{`${b.label} · ${xLabel}=${c.xText}: ${fmtDollars(b.value)}`}</title>
                      </circle>
                    );
                  })}
                  {Number.isFinite(c.st.mean) && (
                    <g stroke={AVG_COLOR}>
                      {Number.isFinite(c.st.std) && (
                        <>
                          <line x1={centerX + 12} x2={centerX + 12}
                                y1={y(c.st.mean + c.st.std)} y2={y(c.st.mean - c.st.std)} strokeWidth="1.5" />
                          <line x1={centerX + 7} x2={centerX + 17}
                                y1={y(c.st.mean + c.st.std)} y2={y(c.st.mean + c.st.std)} strokeWidth="1.5" />
                          <line x1={centerX + 7} x2={centerX + 17}
                                y1={y(c.st.mean - c.st.std)} y2={y(c.st.mean - c.st.std)} strokeWidth="1.5" />
                        </>
                      )}
                      <line x1={centerX + 4} x2={centerX + 20} y1={y(c.st.mean)} y2={y(c.st.mean)} strokeWidth="2.5">
                        <title>{`mean ${fmtDollars(c.st.mean)}${Number.isFinite(c.st.std) ? ` ± ${fmtDollars(c.st.std)}` : ''}`}</title>
                      </line>
                    </g>
                  )}
                </>
              )}
              <text x={centerX} y={H - m.bottom + 18} textAnchor="middle" fontSize="12" fill="#334155"
                    style={{ fontVariantNumeric: 'tabular-nums' }}>
                {c.xText}
              </text>
              <text x={centerX} y={H - m.bottom + 34} textAnchor="middle" fontSize="12"
                    fontWeight="600" fill="#64748b" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {c.num != null ? `(${c.num})` : ''}
              </text>
            </g>
          );
        })}
        <text x={m.left + pw / 2} y={H - 8} textAnchor="middle" fontSize="12" fill="#475569">
          {xLabel}  ·  (n) = sweep number in the list
        </text>
      </svg>
    </div>
  );
}

export default function SweepCompare({ rows = [] }) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState('bars');          // 'bars' | 'dots'
  const [includeRuns, setIncludeRuns] = useState(false);
  const sweeps = useMemo(() => groupSweeps(rows), [rows]);
  const runs = useMemo(() => groupRuns(rows), [rows]);

  // The pickable list: sweeps, optionally with individual Runs mixed in,
  // newest-first.
  const items = useMemo(() => {
    const list = includeRuns ? [...sweeps, ...runs] : sweeps;
    return [...list].sort((a, b) =>
      String(b.sample.ts ?? '').localeCompare(String(a.sample.ts ?? '')));
  }, [sweeps, runs, includeRuns]);

  // Selection: which items to plot. Default to all once opened.
  const [selected, setSelected] = useState(null);   // null = "not yet initialised"
  const sel = selected ?? new Set(items.map(g => g.key));

  // X-axis parameter: default to the first param that actually varies
  // across the selected items.
  const [xParam, setXParam] = useState(null);
  const selectedSweeps = items.filter(g => sel.has(g.key));
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
  // "Recent N" quick-pick — items are listed newest-first, so the N most
  // recently run are the first N keys.
  const [recentN, setRecentN] = useState('7');
  function applyRecent() {
    const n = Math.max(1, Math.min(items.length, Math.round(Number(recentN) || 0)));
    setSelected(new Set(items.slice(0, n).map(g => g.key)));
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

  // Picker number for each item (1-based, in the newest-first list order)
  // so a cluster can be labelled with the same number the checklist shows.
  const sweepNumber = useMemo(() => {
    const m = new Map();
    items.forEach((g, i) => m.set(g.key, i + 1));
    return m;
  }, [items]);

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
        return { key: g.key, num: sweepNumber.get(g.key),
                 xVal: Number(xv), xText: fmtParamVal(xv), bars };
      })
      .sort((a, b) => (Number.isFinite(a.xVal) && Number.isFinite(b.xVal))
        ? a.xVal - b.xVal
        : String(a.xText).localeCompare(String(b.xText)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSweeps, xKey, allSeeds, sweepNumber]);

  // Export the selected items as CSV — one row per cluster (in chart
  // order), a column per seed's profit, plus mean / std and the true-
  // optimum average. Effectively the bar chart as a table.
  function exportCsv() {
    const ordered = [...selectedSweeps].sort((a, b) => {
      const av = Number(a.sample[xKey]), bv = Number(b.sample[xKey]);
      return (Number.isFinite(av) && Number.isFinite(bv))
        ? av - bv
        : fmtParamVal(a.sample[xKey]).localeCompare(fmtParamVal(b.sample[xKey]));
    });
    const header = ['sweep_num', 'when', 'policy', 'horizon', 'repeat', 'N', 'rho_band', 'm_star',
      ...allSeeds.map(s => `seed_${s}`), 'mean_profit', 'std_profit', 'optimal_avg'];
    const lines = [header.map(csvCell).join(',')];
    for (const g of ordered) {
      const s = g.sample;
      const bySeed = new Map(g.seedRows.map(r => [Number(r.seed), Number(r.cumulative)]));
      const seedVals = allSeeds.map(sd => (bySeed.has(sd) ? bySeed.get(sd) : null));
      const st = meanStd(seedVals.filter(Number.isFinite));
      const optimal = g.avgRow && Number.isFinite(Number(g.avgRow.optimal))
        ? Number(g.avgRow.optimal)
        : meanStd(g.seedRows.map(r => Number(r.optimal))).mean;
      const rec = [
        sweepNumber.get(g.key), s.ts, POLICY_SHORT[s.policy] ?? s.policy ?? '',
        s.horizon ?? '', s.repeat ?? '', s.budget ?? '', fmtParamVal(s.rho_ell), s.mstar ?? '',
        ...seedVals.map(round2), round2(st.mean), round2(st.std), round2(optimal),
      ];
      lines.push(rec.map(csvCell).join(','));
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    downloadCsv(`sweep-compare-${stamp}.csv`, lines.join('\r\n'));
  }

  return (
    <div className="card" style={{ padding: '10px 14px', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
          Compare sweeps
        </div>
        <span style={{ fontSize: 12, color: '#64748b' }}>
          {sweeps.length + runs.length === 0
            ? '(run a seed sweep to populate this)'
            : `(${sweeps.length} ${sweeps.length === 1 ? 'sweep' : 'sweeps'}` +
              `${runs.length ? `, ${runs.length} runs` : ''} in the log; profit per seed)`}
        </span>
        <button type="button" onClick={() => setOpen(v => !v)}
                disabled={sweeps.length + runs.length === 0}
                style={{ marginLeft: 'auto', ...btn(false),
                         opacity: sweeps.length + runs.length === 0 ? 0.5 : 1,
                         cursor: sweeps.length + runs.length === 0 ? 'not-allowed' : 'pointer' }}>
          {open ? 'Close' : 'Open'}
        </button>
      </div>

      {open && sweeps.length + runs.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 10 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>
                  Sweeps to plot
                </span>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  ({sel.size} of {items.length})
                </span>
                <button type="button" onClick={() => setSelected(new Set(items.map(g => g.key)))}
                        style={miniBtn} title="Select everything in the list">All</button>
                <button type="button" onClick={() => setSelected(new Set())}
                        style={miniBtn} title="Clear the selection">None</button>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#475569' }}
                      title="Select the N most recently run items (top of the list).">
                  Recent
                  <input type="number" min={1} max={items.length}
                         value={recentN}
                         onChange={e => setRecentN(e.target.value)}
                         onKeyDown={e => { if (e.key === 'Enter') applyRecent(); }}
                         style={{ width: 44, padding: '2px 4px', border: '1px solid #cbd5e1',
                                  borderRadius: 4, fontSize: 12 }} />
                  <button type="button" onClick={applyRecent} style={miniBtn}>Apply</button>
                </span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12,
                                color: '#475569', cursor: 'pointer' }}
                       title="Add individual Run rows (single-seed points) to the list.">
                  <input type="checkbox" checked={includeRuns}
                         onChange={e => { setIncludeRuns(e.target.checked); setSelected(null); }} />
                  Include Runs
                </label>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 168,
                            overflowY: 'auto', paddingRight: 6 }}>
                {items.map((g, i) => (
                  <label key={g.key} style={{ display: 'flex', alignItems: 'center', gap: 6,
                                              fontSize: 12.5, color: '#0f172a', cursor: 'pointer' }}>
                    <input type="checkbox" checked={sel.has(g.key)} onChange={() => toggle(g.key)} />
                    <span style={{ color: '#94a3b8', minWidth: 16, textAlign: 'right' }}>{i + 1}.</span>
                    {sweepLabel(g)}
                    <span style={{ color: '#94a3b8' }}>
                      {g.isRun
                        ? `(run · seed ${g.sample.seed})`
                        : `(${g.seedRows.length} ${g.seedRows.length === 1 ? 'seed' : 'seeds'})`}
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
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>
                View
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button type="button" onClick={() => setMode('bars')} style={btn(mode === 'bars')}
                        title="One bar per seed, plus the average">Bars</button>
                <button type="button" onClick={() => setMode('dots')} style={btn(mode === 'dots')}
                        title="Seed values as dots with a mean ± 1 std bar — scales better for many seeds">
                  Dots ± std
                </button>
              </div>
            </div>
          </div>

          {clusters.length === 0 ? (
            <p style={{ fontSize: 12.5, color: '#64748b' }}>
              Select at least one item to plot.
            </p>
          ) : (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14,
                            marginBottom: 8, fontSize: 12, color: '#475569' }}>
                {allSeeds.map(seed => (
                  <span key={seed} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 11, height: 11,
                                   borderRadius: mode === 'dots' ? '50%' : 2,
                                   background: seedColor(seed) }} />
                    Seed {seed}
                  </span>
                ))}
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  {mode === 'dots' ? (
                    <span style={{ width: 14, height: 0, borderTop: `3px solid ${AVG_COLOR}` }} />
                  ) : (
                    <span style={{ width: 13, height: 13, borderRadius: 2, background: AVG_COLOR }} />
                  )}
                  {mode === 'dots' ? 'mean ± 1 std' : 'Sweep avg'}
                </span>
                <button type="button" onClick={exportCsv}
                        style={{ marginLeft: 'auto', ...miniBtn }}
                        title="Save the selected sweeps (per-seed profit, mean, std, optimal) to a CSV file.">
                  Download CSV
                </button>
              </div>
              <SweepChart clusters={clusters} xLabel={xMeta.label} mode={mode} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
