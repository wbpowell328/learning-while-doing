import { useState, useEffect, useRef } from 'react';
import { cloneWithSeed, runExperiment, getReveal, deleteSession } from '../api';

// localStorage keys for the two sweep-only inputs — persist across
// page reloads so a Render sleep + 404 → auto-recovery reload
// doesn't eat the user's just-typed sweep parameters. Horizon and
// Repeat now mirror the ExperimentBar above (read-only display),
// so their localStorage entries are no longer needed here.
const LS_KEYS = {
  baseSeed: 'lwd_sweep_base_seed_v1',
  nSeeds:   'lwd_sweep_n_seeds_v1',
};
function readLS(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw != null ? raw : fallback;
  } catch (_) { return fallback; }
}
function writeLS(key, value) {
  try { localStorage.setItem(key, String(value)); } catch (_) { /* ignore */ }
}

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
  currentHorizon,   // string — Horizon input value from the ExperimentBar
  currentRepeat,    // string — Repeat input value from the ExperimentBar
  disabled = false, // parent should pass loading || !session
  onSessionLost,    // () => void — called when the backend has forgotten
                    //   the current session (Render sleep/restart) so the
                    //   parent can trigger its 404-recovery flow.
  onLogEntry,       // (entry) => void — feed each finished sweep row
                    //   into the App-level Run history log so the user
                    //   sees the whole sweep alongside their manual runs.
  onSessionSeedChange, // (newSeed) => void — this input doubles as the
                       //   session's random-number seed control; the
                       //   parent mutates the backend session's seed
                       //   and triggers a Restart so the next Run
                       //   starts clean under the new seed.
}) {
  const dim = session?.dim ?? 1;
  const baseSeedDefault = String(Number(session?.session_seed ?? 42));
  const [baseSeed,  setBaseSeed]  = useState(() => readLS(LS_KEYS.baseSeed, baseSeedDefault));
  const [nSeeds,    setNSeeds]    = useState(() => readLS(LS_KEYS.nSeeds,   '5'));
  useEffect(() => { writeLS(LS_KEYS.baseSeed, baseSeed); }, [baseSeed]);
  useEffect(() => { writeLS(LS_KEYS.nSeeds,   nSeeds);   }, [nSeeds]);
  // Auto-sync the sweep's Base seed to the current session's seed
  // whenever a fresh session is created (session.session_seed
  // changes). Fixes the confusion where the top bar and the sweep
  // ran on different seeds — the sweep's first row was never the
  // same as a manual Run. Fires only on session-change, so any
  // manual edits the user makes after that aren't clobbered by
  // unrelated re-renders.
  useEffect(() => {
    const s = session?.session_seed;
    if (s != null && Number.isFinite(Number(s))) {
      setBaseSeed(String(Math.round(Number(s))));
    }
  }, [session?.session_seed]);
  // Horizon and Repeat are read-only mirrors of the ExperimentBar
  // inputs above. Fall back to sensible defaults if the parent
  // hasn't fed them yet (very first render).
  const horizon = currentHorizon ?? '50';
  const repeat  = currentRepeat  ?? '1';

  const [rows, setRows]         = useState([]);   // {seed, ...} rows collected so far
  const [status, setStatus]     = useState(null); // in-progress human label
  const [running, setRunning]   = useState(false);
  const [error, setError]       = useState(null);
  // Show / hide the results table. Warren can Close the report to
  // clear visual space; the rows stay in state so Open restores the
  // same table. A fresh sweep re-opens it automatically.
  const [showReport, setShowReport] = useState(true);
  // Abort plumbing — a fresh controller per sweep; Stop calls
  // .abort() so the in-flight fetch throws AbortError and we can
  // break out of the loop without waiting for the current seed to
  // finish. Ref (not state) because we read it from the click
  // handler, don't need a re-render on change.
  const abortRef = useRef(null);

  const parsedBase   = Number(baseSeed);
  const parsedNSeeds = Math.max(2, Math.min(20, Math.round(Number(nSeeds)  || 0)));
  const parsedHoriz  = Math.max(1, Math.min(500, Math.round(Number(horizon) || 0)));
  const parsedRep    = Math.max(1, Math.min(100, Math.round(Number(repeat) || 0)));
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
        K: Math.min(99, Math.max(0, parsedRep - 1)),
        theta_init: initialTheta,
      };
      const runResp = await runExperiment(clonedId, spec, signal);
      const history = runResp?.history ?? [];
      const totalProfit = history.reduce((s, row) => s + Number(row[1] ?? 0), 0);
      // best_impparam is what the policy would pick next given the
      // belief after all iterations — the natural "where did it land"
      // summary for a converged run.
      const policyTheta = runResp?.best_impparam;
      // Per-seed reveal — n_reps=50 Monte-Carlo estimate of the true
      // reward curve using THIS seed's noise draws. Different seeds
      // give slightly different estimates of the same population
      // optimum; showing them side by side makes the "policy beat the
      // optimum on this seed" surprise visible as MC noise rather
      // than a bug.
      let trueOptTheta = NaN, trueOptProfit = NaN;
      try {
        const rev = await getReveal(clonedId, signal);
        if (rev && Number.isFinite(rev.true_best_impparam)
                && Number.isFinite(rev.true_max_reward_per_day)) {
          trueOptTheta  = rev.true_best_impparam;
          trueOptProfit = rev.true_max_reward_per_day * parsedHoriz * parsedRep;
        }
      } catch (err) {
        if (isAbortError(err)) throw err;   // stop propagates up
        // Any other reveal failure is non-fatal — leave the two
        // columns as "—" so the sweep still yields the policy's
        // achieved results.
      }
      return { seed: seedValue, policyTheta, totalProfit, trueOptTheta, trueOptProfit };
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
  function isSessionLost(err) {
    // Backend clone_with_seed returns 404 "Session not found" when the
    // process has been restarted (Render free-tier sleep) and the
    // in-memory session dict has been wiped.
    return err && /\b404\b.*session.*not found/i.test(String(err.message ?? ''));
  }

  async function handleRunSweep() {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    setRows([]);
    setShowReport(true);   // new sweep → make sure the report is visible
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
          // Also feed this row into the shared Run history log so the
          // user can review sweep entries alongside manual runs. Same
          // shape as the App.jsx logRunResult() call in
          // handleRunExperiment. Action "Sweep" plus the walking seed
          // makes sweep entries easy to spot in the history table.
          if (onLogEntry) {
            const totalDays = parsedHoriz * parsedRep;
            onLogEntry({
              ts: new Date().toISOString(),
              seed: Number(row.seed),
              action: 'Sweep',
              app: session.app_name,
              policy: currentPolicy,
              horizon: parsedHoriz,
              repeat: parsedRep,
              theta_init: initialTheta,
              best_theta: row.policyTheta,
              latest: row.totalProfit,
              cumulative: row.totalProfit,
              total_days: totalDays,
              optimal: Number.isFinite(row.trueOptProfit) ? row.trueOptProfit : null,
            });
          }
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
      } else if (isSessionLost(err) && onSessionLost) {
        // Backend forgot the session (Render sleep). Hand off to the
        // parent's 404-recovery flow (reload with ?auto=1 so a fresh
        // session gets created). Our inputs are already persisted to
        // localStorage, so the user's sweep params survive the reload.
        setStatus('Session expired — starting a new one…');
        onSessionLost();
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

  // Summary stats for the four numeric columns. 1-D only for θ (a
  // 2-D optimum is a vector, mean/std across seeds isn't meaningful
  // component-wise here); profits summarise in both dims.
  const canSummariseTheta = dim === 1 && rows.length >= 2;
  const policyThetaStats = canSummariseTheta
    ? meanStd(rows.map(r => Number(r.policyTheta)))
    : null;
  const profitStats = rows.length >= 2
    ? meanStd(rows.map(r => Number(r.totalProfit)))
    : null;
  const trueThetaStats = canSummariseTheta
    ? meanStd(rows.map(r => Number(r.trueOptTheta)))
    : null;
  const trueProfitStats = rows.length >= 2
    ? meanStd(rows.map(r => Number(r.trueOptProfit)))
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
          belief and the same initial θ, and reveals the per-seed
          true-optimum estimate (n=50 replications) alongside what
          the policy achieved. Sequential — Render free-tier is
          single-worker.
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10,
                    flexWrap: 'wrap', marginTop: 10 }}>
        <span style={labelStyle}
              title="This is the session's random-number seed — the same seed the top bar uses for Run/Repeat and the seed the sweep counts up from. Changing it and pressing Enter (or clicking away) resets the game to a fresh start under the new seed.">
          Seed:
        </span>
        <input type="number" value={baseSeed}
               onChange={e => setBaseSeed(e.target.value)}
               onBlur={() => {
                 // Commit the typed value to the backend session's
                 // seed. handleSessionSeedChange in App.jsx short-
                 // circuits when the value hasn't actually changed,
                 // so simply re-focusing the box is a no-op.
                 if (onSessionSeedChange) onSessionSeedChange(baseSeed);
               }}
               onKeyDown={e => {
                 if (e.key === 'Enter') {
                   e.preventDefault();
                   e.currentTarget.blur();   // triggers the commit above
                 }
               }}
               disabled={running}
               title="Session seed. Enter or click away to apply — resets the game so the next Run starts fresh under the new seed."
               style={boxStyle} />
        <span style={labelStyle}>· Seeds to run:</span>
        <input type="number" value={nSeeds} min={2} max={20}
               onChange={e => setNSeeds(e.target.value)}
               disabled={running}
               style={{ ...boxStyle, width: 46 }} />
        <span style={labelStyle} title="Read-only — mirrors the Horizon box above so the sweep uses the same batch length as your regular Run.">
          · Horizon (days):
        </span>
        <input type="number" value={horizon} disabled readOnly
               tabIndex={-1}
               style={{ ...boxStyle, width: 52, background: '#f1f5f9',
                        color: '#475569', cursor: 'not-allowed' }}
               title="Set this via the Horizon input above the control bar." />
        <span style={labelStyle} title="Read-only — mirrors the Repeat box above so the sweep runs the same number of iterations per seed as your regular Run.">
          · Repeat:
        </span>
        <input type="number" value={repeat} disabled readOnly
               tabIndex={-1}
               style={{ ...boxStyle, width: 46, background: '#f1f5f9',
                        color: '#475569', cursor: 'not-allowed' }}
               title="Set this via the Repeat input above the control bar." />
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
        <div style={{ marginTop: 12,
                      display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 12.5, fontWeight: 600, color: '#334155' }}>
            Results ({rows.length} {rows.length === 1 ? 'seed' : 'seeds'})
          </span>
          <button type="button"
                  onClick={() => setShowReport(v => !v)}
                  style={{ padding: '3px 10px', border: '1px solid #cbd5e1',
                           borderRadius: 4, fontSize: 12, background: '#fff',
                           cursor: 'pointer' }}
                  title={showReport
                    ? 'Hide the results table (rows stay saved)'
                    : 'Show the results table again'}>
            {showReport ? 'Close' : 'Open'}
          </button>
        </div>
      )}

      {rows.length > 0 && showReport && (
        <div style={{ marginTop: 8, overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%',
                          maxWidth: 720 }}>
            <thead>
              <tr>
                <th style={thStyle}>Seed</th>
                <th style={{ ...thStyle, textAlign: 'right' }}
                    title="Where the policy ended up after all iterations.">Policy θ</th>
                <th style={{ ...thStyle, textAlign: 'right' }}
                    title="Cumulative profit the policy earned on this seed's noise draws.">Policy profit</th>
                <th style={{ ...thStyle, textAlign: 'right' }}
                    title="Monte-Carlo estimate of the true argmax, using n=50 replications seeded from this run.">True opt θ</th>
                <th style={{ ...thStyle, textAlign: 'right' }}
                    title="MC-estimated true reward per day × Horizon × Repeat. What a perfect θ picker would have earned in expectation.">True opt profit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.seed}-${i}`}>
                  <td style={tdStyle}>{r.seed}</td>
                  <td style={tdNum}>
                    {dim === 1
                      ? fmt(r.policyTheta, 3)
                      : Array.isArray(r.policyTheta)
                        ? `(${r.policyTheta.map(x => fmt(x, 3)).join(', ')})`
                        : '—'}
                  </td>
                  <td style={tdNum}>{fmtDollars(r.totalProfit)}</td>
                  <td style={tdNum}>
                    {dim === 1 ? fmt(r.trueOptTheta, 3) : '—'}
                  </td>
                  <td style={tdNum}>{fmtDollars(r.trueOptProfit)}</td>
                </tr>
              ))}
              {(policyThetaStats || profitStats || trueThetaStats || trueProfitStats) && (
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>mean ± std</td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {policyThetaStats
                      ? `${fmt(policyThetaStats.mean, 3)} ± ${fmt(policyThetaStats.std, 3)}`
                      : '—'}
                  </td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {profitStats
                      ? `${fmtDollars(profitStats.mean)} ± ${fmtDollars(profitStats.std)}`
                      : '—'}
                  </td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {trueThetaStats
                      ? `${fmt(trueThetaStats.mean, 3)} ± ${fmt(trueThetaStats.std, 3)}`
                      : '—'}
                  </td>
                  <td style={{ ...tdNum, fontWeight: 600 }}>
                    {trueProfitStats
                      ? `${fmtDollars(trueProfitStats.mean)} ± ${fmtDollars(trueProfitStats.std)}`
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
