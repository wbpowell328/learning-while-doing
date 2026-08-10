import { useState } from 'react';

// Run history log — a lightweight, cross-session record of each
// Run / One-more click so Warren can review earlier simulations
// without having to screenshot the score boxes every time. Rows
// persist to localStorage so they survive page reloads. Storage cap
// is enforced in the parent (App.jsx).

const POLICY_SHORT = {
  okg_ryzhov:        'Ryzhov',
  okg:               'Online KG',
  okg_indep:         'Online KG (indep)',
  kg:                'Offline KG',
  kg_indep:          'Offline KG (indep)',
  ie:                'IE',
  ie_15:             'IE (1.5)',
  greedy:            'Greedy',
  randomized_greedy: 'Randomized greedy',
  random:            'Random',
  human:             'Manual',
};
const APP_SHORT = {
  cash_balance:    '1-D',
  cash_balance_2d: '2-D',
};

const thStyle = {
  textAlign: 'left', padding: '4px 8px', fontSize: 12,
  color: '#334155', borderBottom: '1px solid #cbd5e1', fontWeight: 600,
  whiteSpace: 'nowrap',
};
const tdStyle = {
  padding: '4px 8px', fontSize: 12.5, color: '#0f172a',
  borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap',
};
const tdNum = { ...tdStyle, textAlign: 'right', fontVariantNumeric: 'tabular-nums' };

function fmtTheta(t) {
  if (t == null) return '—';
  if (Array.isArray(t)) return `(${t.map(x => Number(x).toFixed(3)).join(', ')})`;
  const n = Number(t);
  return Number.isFinite(n) ? n.toFixed(3) : '—';
}
function fmtDollars(v) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  const abs = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}
function fmtTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mi = String(d.getMinutes()).padStart(2, '0');
    return `${mm}-${dd} ${hh}:${mi}`;
  } catch (_) { return '—'; }
}

export default function RunHistory({ rows = [], onClear }) {
  // Default collapsed so it doesn't dominate the page for a brand-new
  // visitor with a fresh log — expand once the user wants to review.
  const [open, setOpen] = useState(false);

  return (
    <div className="card" style={{ padding: '10px 14px', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
          Run history
        </div>
        <span style={{ fontSize: 12, color: '#64748b' }}>
          {rows.length === 0
            ? '(no runs yet — hit Run to start logging)'
            : `(${rows.length} ${rows.length === 1 ? 'entry' : 'entries'}; newest first, persists across reloads)`}
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button type="button"
                  onClick={() => setOpen(v => !v)}
                  disabled={rows.length === 0}
                  style={{ padding: '3px 10px', border: '1px solid #cbd5e1',
                           borderRadius: 4, fontSize: 12, background: '#fff',
                           cursor: rows.length === 0 ? 'not-allowed' : 'pointer',
                           opacity: rows.length === 0 ? 0.5 : 1 }}
                  title={open ? 'Hide the table (entries stay saved)' : 'Show the table'}>
            {open ? 'Close' : 'Open'}
          </button>
          <button type="button"
                  onClick={onClear}
                  disabled={rows.length === 0}
                  style={{ padding: '3px 10px', border: '1px solid #cbd5e1',
                           borderRadius: 4, fontSize: 12, background: '#fff',
                           color: '#b91c1c',
                           cursor: rows.length === 0 ? 'not-allowed' : 'pointer',
                           opacity: rows.length === 0 ? 0.5 : 1 }}
                  title="Delete every logged entry. Cannot be undone.">
            Clear
          </button>
        </div>
      </div>

      {open && rows.length > 0 && (
        <div style={{ marginTop: 10, overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%' }}>
            <thead>
              <tr>
                <th style={thStyle}>When</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Random-number seed for the session that generated this row. Seed-sweep entries walk this column by 1.">Seed</th>
                <th style={thStyle}>App</th>
                <th style={thStyle}>Policy</th>
                <th style={thStyle}>Action</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Days per iteration">Horiz</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Iterations in this click">Rep</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="θ used for the first iteration (Run) or θ picked by the policy on the last iteration (One more)">θ used</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Policy's belief argmax after this action">Best θ</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Reward from this action alone">Latest</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Cumulative reward since the last Restart">Cumul</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="What a perfect θ picker would have earned over the same total days">Optimal</th>
                <th style={{ ...thStyle, textAlign: 'right' }} title="Total days simulated since the last Restart">Days</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.ts}-${i}`}>
                  <td style={tdStyle}>{fmtTime(r.ts)}</td>
                  <td style={tdNum}>{Number.isFinite(r.seed) ? r.seed : '—'}</td>
                  <td style={tdStyle}>{APP_SHORT[r.app] ?? r.app ?? '—'}</td>
                  <td style={tdStyle}>{POLICY_SHORT[r.policy] ?? r.policy ?? '—'}</td>
                  <td style={tdStyle}>{r.action ?? '—'}</td>
                  <td style={tdNum}>{r.horizon ?? '—'}</td>
                  <td style={tdNum}>{r.repeat ?? '—'}</td>
                  <td style={tdNum}>{fmtTheta(r.theta_init)}</td>
                  <td style={tdNum}>{fmtTheta(r.best_theta)}</td>
                  <td style={tdNum}>{fmtDollars(r.latest)}</td>
                  <td style={tdNum}>{fmtDollars(r.cumulative)}</td>
                  <td style={tdNum}>{fmtDollars(r.optimal)}</td>
                  <td style={tdNum}>{r.total_days ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
