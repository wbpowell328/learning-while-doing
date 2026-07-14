export default function HistoryTable({ history }) {
  if (history.length === 0) {
    return <p style={{ color: '#94a3b8', fontSize: 14, padding: '12px 0' }}>No observations yet.</p>;
  }

  return (
    <div className="table-scroll">
      <table className="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>C*</th>
            <th>Total cost</th>
          </tr>
        </thead>
        <tbody>
          {[...history].reverse().map(([c, cost], i) => (
            <tr key={i}>
              <td style={{ color: '#94a3b8' }}>{history.length - i}</td>
              <td><code>{c.toFixed(4)}</code></td>
              <td>${cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
