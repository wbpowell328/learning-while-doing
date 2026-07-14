function formatDollar(v) {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1000) return `$${(v / 1000).toFixed(0)}k`;
  return `$${v.toFixed(0)}`;
}

const TH = {
  textAlign: 'left',
  padding: '5px 10px',
  fontSize: 10,
  fontWeight: 700,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  color: '#64748b',
  borderBottom: '1px solid #e2e8f0',
  background: 'white',
  position: 'sticky',
  top: 0,
};

const TD = { padding: '5px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12 };

export default function JumpLog({ events, initialAum }) {
  if (!events || events.length === 0) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 8 }}>
        Institutional events this run ({events.length})
      </div>
      <div style={{ maxHeight: 180, overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: 6 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={TH}>Week</th>
              <th style={TH}>Day</th>
              <th style={TH}>Type</th>
              <th style={TH}>Size (% AUM)</th>
              <th style={TH}>Approx. $</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => {
              const isOut = e.direction === -1;
              const approxDollars = e.size_fraction * initialAum;
              return (
                <tr key={i} style={{ background: isOut ? '#fef2f2' : '#f0fdf4' }}>
                  <td style={TD}>{Math.floor(e.day / 5)}</td>
                  <td style={TD}>{e.day}</td>
                  <td style={{ ...TD, fontWeight: 600, color: isOut ? '#dc2626' : '#16a34a' }}>
                    {isOut ? 'Redemption' : 'Subscription'}
                  </td>
                  <td style={TD}>{(e.size_fraction * 100).toFixed(1)}%</td>
                  <td style={{ ...TD, fontFamily: 'monospace' }}>
                    {isOut ? '−' : '+'}{formatDollar(approxDollars)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
