// A θ slider whose track lines up under the KG chart's plot area.
//
// The KG chart's viewBox is 720 wide with PAD.left=72 and PAD.right=96,
// so its plot area sits at 10.00%–86.67% of the container width. This
// component uses matching padding so the slider handle points directly
// at the corresponding x value on the KG curves above.
//
// Keep these percentages in sync with KGChart.jsx PAD.

const PAD_LEFT_PCT  = (72 / 720) * 100; // 10.00%
const PAD_RIGHT_PCT = (96 / 720) * 100; // 13.33%
const C_MIN = 0.01, C_MAX = 0.20;

export default function ImpparamSlider({ impparam, setImpparam, onRun, loading, exhausted }) {
  return (
    <div style={{ paddingLeft: `${PAD_LEFT_PCT}%`, paddingRight: `${PAD_RIGHT_PCT}%` }}>
      {/* Slider — sits directly under the KG chart's x-axis */}
      <input
        type="range"
        min={C_MIN} max={C_MAX} step={0.001}
        value={impparam}
        onChange={(e) => setImpparam(Number(e.target.value))}
        disabled={exhausted}
        style={{
          width: '100%',
          margin: 0,
          accentColor: '#2563eb',
          cursor: exhausted ? 'not-allowed' : 'pointer',
          display: 'block',
        }}
      />

      {/* Tick labels matching the KG chart's x-axis ticks */}
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 11, color: '#94a3b8', marginTop: 2,
      }}>
        {[0.01, 0.05, 0.10, 0.15, 0.20].map((v) => {
          const left = ((v - C_MIN) / (C_MAX - C_MIN)) * 100;
          return (
            <span key={v} style={{
              position: 'absolute', transform: 'translateX(-50%)',
              left: `calc(${PAD_LEFT_PCT}% + (${100 - PAD_LEFT_PCT - PAD_RIGHT_PCT}% * ${left / 100}))`,
            }}>
              {v.toFixed(2)}
            </span>
          );
        })}
        {/* Reserve vertical space so the absolutely-positioned tick labels don't overlap the row below */}
        <span style={{ visibility: 'hidden' }}>0.00</span>
      </div>

      {/* Number input + Run button */}
      <div style={{
        display: 'flex', gap: 12, marginTop: 14, alignItems: 'center',
      }}>
        <label style={{ fontSize: 13, color: '#374151', fontWeight: 600 }}>θ =</label>
        <input
          type="number"
          value={impparam.toFixed(3)}
          min={C_MIN} max={C_MAX} step={0.001}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (v >= C_MIN && v <= C_MAX) setImpparam(v);
          }}
          disabled={exhausted}
          style={{
            width: 90, padding: '4px 8px', border: '1px solid #e2e8f0',
            borderRadius: 6, fontSize: 14, fontWeight: 700,
            textAlign: 'center', fontFamily: 'monospace',
            background: exhausted ? '#f8fafc' : 'white',
          }}
        />
        {exhausted ? (
          <div style={{
            flex: 1, textAlign: 'center', fontSize: 13, color: '#dc2626',
            padding: '8px 12px', background: '#fef2f2', borderRadius: 6,
          }}>
            Budget exhausted — start a new session
          </div>
        ) : (
          <button
            className="btn btn-primary"
            onClick={onRun}
            disabled={loading}
            style={{ flex: 1, padding: '10px', fontSize: '0.95rem' }}
          >
            {loading ? 'Running simulation…' : `Run simulation at θ = ${impparam.toFixed(3)} →`}
          </button>
        )}
      </div>
    </div>
  );
}
