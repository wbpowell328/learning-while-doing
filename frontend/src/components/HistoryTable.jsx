// Observation history with per-row μ^n and offline analytic correlated KG,
// evaluated at the belief state BEFORE that row's observation was folded in
// (i.e. what the policy saw when it decided to pick that θ). The μ and KG
// columns appear only when the caller passes `enrichedRows` from the
// /observations_enriched endpoint; otherwise the table falls back to the
// original three-column view.

export default function HistoryTable({ history, enrichedRows }) {
  if (history.length === 0) {
    return <p style={{ color: '#94a3b8', fontSize: 14, padding: '12px 0' }}>No observations yet.</p>;
  }

  const fmtTheta = (c) => {
    if (Array.isArray(c)) return `(${c.map(v => Number(v).toFixed(3)).join(', ')})`;
    return Number(c).toFixed(4);
  };
  const fmtDollars = (v) =>
    `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  const fmtDollarsPrecise = (v) => {
    const a = Math.abs(v);
    if (a >= 1000) return `$${(v / 1000).toFixed(2)}k`;
    if (a >= 10)   return `$${v.toFixed(0)}`;
    return `$${v.toFixed(2)}`;
  };

  // Prefer the enriched payload when it exists AND matches history length.
  // If they get out of sync (mid-fetch), fall back to plain history so we
  // never render stale μ/KG next to a fresh observation.
  const useEnriched = Array.isArray(enrichedRows) &&
                      enrichedRows.length === history.length;

  return (
    <div className="table-scroll">
      <table className="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>θ</th>
            <th>Total reward</th>
            {useEnriched && <th style={{ textTransform: 'none' }}
              title="Posterior mean at this θ under the belief state BEFORE this step's observation was folded in — what the policy saw when it decided">
              μ<sup>n</sup>
            </th>}
            {useEnriched && <th style={{ textTransform: 'none' }}
              title="Offline analytic correlated KG at this θ under the same pre-step belief state; the info-value estimate that drove the policy's pick">
              KG
            </th>}
          </tr>
        </thead>
        <tbody>
          {[...history].map((_row, i) => i).reverse().map((origIdx) => {
            const [c, reward] = history[origIdx];
            const enriched = useEnriched ? enrichedRows[origIdx] : null;
            return (
              <tr key={origIdx}>
                <td style={{ color: '#94a3b8' }}>{origIdx + 1}</td>
                <td><code>{fmtTheta(c)}</code></td>
                <td>{fmtDollars(reward)}</td>
                {useEnriched && <td>{fmtDollars(enriched?.mu ?? 0)}</td>}
                {useEnriched && <td>{fmtDollarsPrecise(enriched?.kg ?? 0)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
