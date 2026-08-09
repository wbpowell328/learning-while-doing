import { useState, useRef } from 'react';
import { cloneWithSeed, runExperiment, deleteSession } from '../api';

// Small "seed variability" panel. The user picks how many independent
// replications to run (each with a fresh session seeded base_seed + i,
// same everything else) and the panel:
//   1. clones the current session with a new seed,
//   2. runs Horizon × Repeat days of the current policy on that clone,
//   3. records optimal θ and cumulative profit,
//   4. deletes the clone,
//   5. loops to the next seed.
// Runs are strictly sequential — Render's free-tier is single-worker,
// so pretending to parallelise wouldn't buy anything and adds ordering
// complexity to the results table.

const boxStyle = {
  padding: '4px 6px', border: '1px solid #cbd5e1', borderRadius: 4,
  fontSize: 13, width: 60, boxSizing: 'border-box',
};
const labelStyle = { color: '#475569', fontSize: 13 };
const thStyle = {
  textAlign: 'left', padding: '4px 8px', fontSize: 12,
  color: '#334155', borderBottom: '1px solid #cbd5e1', fontWeight: 600,
};
const tdStyle = {
  padding: '4px 8px', fontSize: 12.5, color: '#0f172a',
  borderBottom: '1px solid #f1f5f9',
};
const tdNum = { ...tdStyle, textAlign: 'right', fontVariantNumeric: 'tabular-nums' };

function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return '—';
  return Number(v).toFixed(digits);
}
function fmtDollars(v) {
  if (!Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '−' : '';
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}
function meanStd(xs) {
  const good = xs.filter(Number.isFinite);
  if (good.length === 0) return { mean: NaN, std: NaN };
  const mean = good.reduce((a, b) => a + b, 0) / good.length;
  if (good.length < 2) return { mean, std: NaN };
  const v = good.reduce((s, x) => s + (x - mean) ** 2, 0) / (good.length - 1);
  return { mean, std: Math.sqrt(v) };
}

export default function SeedSweep({
  session,          // { id, session_seed, initial_theta, dim, ... }
  currentPolicy,    // effective policy value in the ExperimentBar right now
  disabled = false, // parent should pass loading || !session
}) {
  const dim = session?.dim ?? 1;
  const baseSeedDefault = Number(session?.session_seed ?? 42);
  const [baseSeed,  setBaseSeed]  = useState(String(baseSeedDefault));
  const [nSeeds,    setNSeeds]    = useState('5');
  const [horizon,   setHorizon]   = useState('50');
  const [repeat,    setRepeat]    = useState('20');

  const [rows, setRows]         = useState([]);   // {seed, optimalTheta, totalProfit}
  const [status, setStatus]     = useState(null); // in-progress human label
  const [running, setRunning]   = useState(false);
  const [error, setError]       = useState(null);
  // Abort plumbing — a fresh controller per sweep; Stop calls
  // .abort() so the in-flight fetch throws AbortError and we can
  // break out of the loop without waiting for the current seed to
  // finish. Ref (not state) because we read it from the click
  // handler, don't need a re-render on change.
  const abortRef = useRef(null);

  const parsedBase   = Number(baseSeed);
  const parsedNSeeds = Math.max(2, Math.min(20, Math.round(Number(nSeeds)  || 0)));
  const parsedHoriz  = Math.max(1, Math.min(500, Math.round(Number(horizon) || 0)));
  const parsedRep    = Math.max(1, Math.min(50, Math.round(Number(repeat) || 0)));
  const canRun = !running && !disabled && Number.isFinite(parsedBase);

  async function runOneSeed(seedValue, initialTheta, signal) {
    let clonedId = null;
    try {
      const resp = await cloneWithSeed(session.id, seedValue, signal);
      clonedId = resp?.session_id;
      if (!clonedId) throw new Error('clone_with_seed returned no session_id');
      // K = extras after iter 1 (backend contract); user-facing Repeat
      // is the total iteration count, matching ExperimentBar.
      const spec = {
        n_days: parsedHoriz,
        policy: currentPolicy,
        K: Math.min(19, Math.max(0, parsedRep - 1)),
        theta_init: initialTheta,
      };
      const runResp = await runExperiment(clonedId, spec, signal);
      const history = runResp?.history ?? [];
      const totalProfit = history.reduce((s, row) => s + Number(row[1] ?? 0), 0);
      // best_impparam is what the policy would pick next given the
      // belief after all iterations — the natural "where did it land"
      // summary for a converged run.
      const optimalTheta = runResp?.best_impparam;
      return { seed: seedValue, optimalTheta, totalProfit };
    } finally {
      // Always clean up the temp session even if we were aborted
      // mid-flight. Not passing `signal` here so the DELETE goes
      // through — otherwise the aborted controller would nuke it too.
      if (clonedId) { try { await deleteSession(clonedId); } catch (_) { /* ignore */ } }
    }
  }

  function isAbortError(err) {
    return err && (err.name === 'AbortError' || /aborted/i.test(String(err.message ?? '')));
  }

  async function handleRunSweep() {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    setRows([]);
    const controller = new AbortController();
    abortRef.current = controller;
    // Same starting θ every run — matches the ExperimentBar's normal
    // "Run" behaviour, so the sweep is truly measuring the same setup
    // under different noise.
    const initialTheta = Array.isArray(session.initial_theta)
      ? session.initial_theta.slice(0, dim)
      : Number(session.initial_theta ?? 0.1);
    try {
      const collected = [];
      for (let i = 0; i < parsedNSeeds; i++) {
        if (controller.signal.aborted) break;
        const seed = Math.round(parsedBase) + i;
        setStatus(`Running ${i + 1} of ${parsedNSeeds} · seed = ${seed}`);
        try {
          const row = await runOneSeed(seed, initialTheta, controller.signal);
          collected.push(row);
          setRows([...collected]);
        } catch (err) {
          if (isAbortError(err)) break;
          throw err;
        }
      }
      const stopped = controller.signal.aborted;
      setStatus(stopped
        ? `Stopped after ${collected.length} of ${parsedNSeeds} · ${collected.length} results kept.`
        : null);
    } catch (err) {
      if (isAbortError(err)) {
        setStatus('Stopped.');
      } else {
        setError(String(err?.message ?? err));
        setStatus(null);
      }
    } finally {
      abortRef.current = null;
      setRunning(false);
    }
  }

  function handleStop() {
    const c = abortRef.current;
    if (c && !c.signal.aborted) {
      setStatus('Stopping…');
      c.abort();
    }
  }

  // Summary stats for the two output columns. 1-D only (2-D optimal
  // theta is a vector, so a scalar mean/std would be misleading).
  const canSummariseTheta = dim === 1 && rows.length >= 2;
  const thetaStats  = canSummariseTheta
    ? meanStd(rows.map(r => Number(r.optimalTheta)))
    : null;
  const profitStats = rows.length >= 2
    ? meanStd(rows.map(r => Number(r.totalProfit)))
    : null;

  return (
    <div className="card" style={{ padding: '12px 16px', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'baseline',
                    justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
          Seed variability sweep
        </div>
        <div style={{ fontSize: 11.5, color: '#64748b', maxWidth: 520 }}>
          Runs the current policy N times on independent noise draws
          (seed = base, base + 1, …), each starting from a fresh
          belief and the same initial θ. Sequential — Render free-tier
          is single-worker.
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10,
                    flexWrap: 'wrap', marginTop: 10 }}>
        <span style={labelStyle}>Base seed:</span>
        <input type="number" value={baseSeed}
               onChange={e => setBaseSeed(e.target.value)}
               disabled={running}
               style={boxStyle} />
        <span style={labelStyle}>· Seeds to run:</span>
        <input type="number" value={nSeeds} min={2} max={20}
               onChange={e => setNSeeds(e.target.value)}
               disabled={running}
               style={{ ...boxStyle, width: 46 }} />
        <span style={labelStyle}>· Horizon (days):</span>
        <input type="number" value={horizon} min={1} max={500}
               onChange={e => setHorizon(e.target.value)}
               disabled={running}
               style={{ ...boxStyle, width: 52 }} />
        <span style={labelStyle}>· Repeat:</span>
        <input type="number" value={repeat} min={1} max={50}
               onChange={e => setRepeat(e.target.value)}
               disabled={running}
               style={{ ...boxStyle, width: 46 }} />
        <button className="btn btn-primary"
                onClick={handleRunSweep}
                disabled={!canRun}
                style={{ padding: '5px 14px', fontSize: 13 }}
                title="Run the current policy on each seed sequentially.">
          {running ? 'Running…' : 'Run seed sweep'}
        </button>
        {running && (
          <button className="btn btn-outline"
                  onClick={handleStop}
                  style={{ padding: '5px 14px', fontSize: 13,
                           borderColor: '#b91c1c', color: '#b91c1c' }}
                  title="Stop the sweep. Keeps whatever rows have already completed.">
            Stop
          </button>
        )}
        {status && (
          <span style={{ fontSize: 12, color: '#0369a1' }}>{status}</span>
        )}
      </div>

      {error && (
        <p style={{ color: '#b91c1c', fontSize: 12, margin: '8px 0 0 0' }}>
          {error}
        </p>
      )}

      {rows.length > 0 && (
        <div style={{ marginTop: 12, overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%',
                          maxWidth: 480 }}>
            <thead>
              <tr>
                <th style={thStyle}>Seed</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Optimal θ</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Total profit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.seed}-${i}`}>
                  <td style={tdStyle}>{r.seed}</td>
                  <td style={tdNum}>
                    {dim === 1
                      ? fmt(r.optimalTheta, 3)
                      : Array.isArray(r.optimalTheta)
                        ? `(${r.optimalTheta.map(x => fmt(x, 3)).join(', ')})`
                        : '—'}
                  </td>
                  <td style={tdNum}>{fmtDollars(r.totalProfit)}</td>
                </tr>
              ))}
              {(thetaStats || profitStats) && (
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>mean ± std</td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {thetaStats
                      ? `${fmt(thetaStats.mean, 3)} ± ${fmt(thetaStats.std, 3)}`
                      : '—'}
                  </td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {profitStats
                      ? `${fmtDollars(profitStats.mean)} ± ${fmtDollars(profitStats.std)}`
                      : '—'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
