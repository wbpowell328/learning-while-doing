export default function HistoryTable({ history }) {
  if (history.length === 0) {
    return <p style={{ color: '#94a3b8', fontSize: 14, padding: '12px 0' }}>No observations yet.</p>;
  }

  // Formats θ as either a scalar or an [a, b] pair.
  const fmtTheta = (c) => {
    if (Array.isArray(c)) return `(${c.map(v => v.toFixed(3)).join(', ')})`;
    return Number(c).toFixed(4);
  };

  return (
    <div className="table-scroll">
      <table className="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>θ</th>
            <th>Total cost</th>
          </tr>
        </thead>
        <tbody>
          {[...history].reverse().map(([c, cost], i) => (
            <tr key={i}>
              <td style={{ color: '#94a3b8' }}>{history.length - i}</td>
              <td><code>{fmtTheta(c)}</code></td>
              <td>${cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
