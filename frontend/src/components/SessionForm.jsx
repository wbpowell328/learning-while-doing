import { useState } from 'react';

export default function SessionForm({ onCreate, error }) {
  const [policy, setPolicy]         = useState('human');
  const [seed, setSeed]             = useState(42);
  const [horizon, setHorizon]       = useState(26);
  const [stationary, setStationary] = useState(true);
  const [budget, setBudget]         = useState(10);
  const [loading, setLoading]       = useState(false);

  const isHuman = policy === 'human';

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    try {
      await onCreate({
        policy,
        session_seed: seed,
        sim_config: { stationary },
        session_config: { horizon_weeks: horizon },
        budget: isHuman ? budget : null,
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="setup-wrap">
      <div className="card setup-card">
        <h1 className="form-title">Learning While Doing</h1>
        <p className="form-subtitle">
          Find the optimal cash buffer ratio C* for a mutual fund.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Mode</label>
            <select value={policy} onChange={e => setPolicy(e.target.value)}>
              <option value="human">Human — I pick C* each round</option>
              <option value="kg">KG — Knowledge Gradient (automated)</option>
              <option value="ie">IE — Max uncertainty exploration (automated)</option>
              <option value="random">Random — baseline (automated)</option>
            </select>
          </div>

          {isHuman && (
            <div className="form-group">
              <label>Adjustment budget</label>
              <input
                type="number" value={budget} min={1} max={50}
                onChange={e => setBudget(Number(e.target.value))}
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                Number of times you can run the simulator
              </span>
            </div>
          )}

          <div className="form-row">
            <div className="form-group">
              <label>Random seed</label>
              <input type="number" value={seed} min={0} max={9999}
                onChange={e => setSeed(Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Weeks per run</label>
              <input type="number" value={horizon} min={1} max={52}
                onChange={e => setHorizon(Number(e.target.value))} />
            </div>
          </div>

          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <input type="checkbox" id="stationary" checked={stationary}
              onChange={e => setStationary(e.target.checked)}
              style={{ width: 16, height: 16, cursor: 'pointer' }} />
            <label htmlFor="stationary" style={{ cursor: 'pointer' }}>
              Stationary regime (no market regime switching)
            </label>
          </div>

          {error && (
            <p style={{ color: '#dc2626', fontSize: 13, marginBottom: 12 }}>{error}</p>
          )}

          <button type="submit" className="btn btn-primary" disabled={loading}
            style={{ width: '100%', padding: '11px', fontSize: '0.95rem' }}>
            {loading ? 'Creating…' : 'Start →'}
          </button>
        </form>
      </div>
    </div>
  );
}
