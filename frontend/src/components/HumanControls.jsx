import { useState } from 'react';

const C_MIN = 0.01;
const C_MAX = 0.20;

export default function HumanControls({ onEvaluate, loading, budget, used }) {
  const [cStar, setCStar] = useState(0.10);

  const remaining = budget - used;
  const exhausted  = remaining <= 0;

  function handleSlider(e) {
    setCStar(Number(e.target.value));
  }

  function handleInput(e) {
    const v = Number(e.target.value);
    if (v >= C_MIN && v <= C_MAX) setCStar(v);
  }

  function handleRun() {
    if (!loading && !exhausted) onEvaluate(cStar);
  }

  const pct = ((cStar - C_MIN) / (C_MAX - C_MIN)) * 100;

  return (
    <div>
      {/* Budget bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
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

      {/* Budget progress bar */}
      <div style={{
        height: 6, background: '#e2e8f0', borderRadius: 9999, marginBottom: 20, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${(remaining / budget) * 100}%`,
          background: exhausted ? '#dc2626' : remaining <= 3 ? '#d97706' : '#16a34a',
          borderRadius: 9999,
          transition: 'width 0.3s ease',
        }} />
      </div>

      {/* C* picker */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
            Set C* (cash buffer ratio)
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="number"
              value={cStar.toFixed(3)}
              min={C_MIN} max={C_MAX} step={0.001}
              onChange={handleInput}
              disabled={exhausted}
              style={{
                width: 80, padding: '4px 8px', border: '1px solid #e2e8f0',
                borderRadius: 6, fontSize: 14, fontWeight: 700,
                textAlign: 'center', fontFamily: 'monospace',
                background: exhausted ? '#f8fafc' : 'white',
              }}
            />
          </div>
        </div>

        <div style={{ position: 'relative', padding: '4px 0' }}>
          {/* Track labels */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
            <span>{C_MIN.toFixed(2)}</span>
            <span>{C_MAX.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={C_MIN} max={C_MAX} step={0.001}
            value={cStar}
            onChange={handleSlider}
            disabled={exhausted}
            style={{ width: '100%', accentColor: '#2563eb', cursor: exhausted ? 'not-allowed' : 'pointer' }}
          />
        </div>

        <div style={{ textAlign: 'center', fontSize: 12, color: '#64748b', marginTop: 4 }}>
          {(pct).toFixed(0)}% of the way across the domain
        </div>
      </div>

      {exhausted ? (
        <div style={{
          padding: '12px 16px', background: '#fef2f2', border: '1px solid #fecaca',
          borderRadius: 8, fontSize: 13, color: '#dc2626', fontWeight: 500, textAlign: 'center',
        }}>
          Budget exhausted — session complete. Start a new session to try again.
        </div>
      ) : (
        <button
          className="btn btn-primary"
          onClick={handleRun}
          disabled={loading}
          style={{ width: '100%', padding: '11px', fontSize: '0.95rem' }}
        >
          {loading ? 'Running simulation…' : `Run simulation at C* = ${cStar.toFixed(3)} →`}
        </button>
      )}
    </div>
  );
}
