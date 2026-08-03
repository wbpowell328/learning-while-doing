// Slim budget-remaining bar for Human mode. The θ slider + Run button
// now live in the KG chart card so the slider aligns with the KG plot's
// x-axis (see CStarSlider.jsx).

export default function HumanControls({ budget, used }) {
  const remaining = budget - used;
  const exhausted = remaining <= 0;

  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 10,
      }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
          Adjustment budget
        </span>
        <span style={{
          fontSize: 13, fontWeight: 700,
          color: exhausted ? '#dc2626' : remaining <= 3 ? '#d97706' : '#16a34a',
        }}>
          {remaining} of {budget} remaining
        </span>
      </div>
      <div style={{
        height: 6, background: '#e2e8f0', borderRadius: 9999, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${(remaining / budget) * 100}%`,
          background: exhausted ? '#dc2626' : remaining <= 3 ? '#d97706' : '#16a34a',
          borderRadius: 9999,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  );
}
