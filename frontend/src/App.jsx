import { useState, useCallback, useRef } from 'react';
import { createSession, runStep, evaluateC, getPosterior, getReveal, getKGComparison, deleteSession, runBatch } from './api';
import SessionForm from './components/SessionForm';
import PosteriorChart from './components/PosteriorChart';
import KGChart from './components/KGChart';
import CStarSlider from './components/CStarSlider';
import HistoryTable from './components/HistoryTable';
import HumanControls from './components/HumanControls';
import CashChart from './components/CashChart';
import JumpLog from './components/JumpLog';
import RevealPanel from './components/RevealPanel';
import BatchResults from './components/BatchResults';
import './App.css';

const POLICY_LABEL = { random: 'Random', ie: 'IE', kg: 'KG', human: 'Human' };

export default function App() {
  const [session,       setSession]       = useState(null);
  const [posterior,     setPosterior]     = useState(null);
  const [kgComparison,  setKgComparison]  = useState(null);
  const [history,       setHistory]       = useState([]);
  const [nSteps,        setNSteps]        = useState(0);
  const [bestCStar,     setBestCStar]     = useState(null);
  const [lastResult,    setLastResult]    = useState(null);
  const [reveal,        setReveal]        = useState(null);
  const [revealLoading, setRevealLoading] = useState(false);
  const [loading,       setLoading]       = useState(false);
  const [autoCount,     setAutoCount]     = useState(0);
  const [error,         setError]         = useState(null);
  const [cStar,         setCStar]         = useState(0.10);
  const [batchResult,   setBatchResult]   = useState(null);
  const [batchProgress, setBatchProgress] = useState(null);
  const stopRef = useRef(false);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const applyResult = useCallback((result, post, kg) => {
    setPosterior(post);
    setKgComparison(kg);
    setBestCStar(result.best_c_star);
    setNSteps(result.n_steps);
    setHistory(prev => [...prev, [result.c_star, result.total_cost]]);
    setLastResult(result);
  }, []);

  const fetchReveal = useCallback(async (sid) => {
    setRevealLoading(true);
    try {
      const rev = await getReveal(sid);
      setReveal(rev);
    } catch (e) {
      setError(String(e));
    } finally {
      setRevealLoading(false);
    }
  }, []);

  // ── Create session ────────────────────────────────────────────────────────

  const handleCreate = useCallback(async ({ policy, session_seed, sim_config, session_config, budget, family, sims_per_policy }) => {
    setError(null);

    // Batch mode: stream the whole family, show progress, then BatchResults.
    if (family) {
      setBatchProgress({ started: true, completed: 0, total: null, current_policy: null, family });
      try {
        const result = await runBatch(
          {
            family,
            sims_per_policy,
            budget: budget ?? 10,
            sim_config,
            session_config,
            session_seed,
          },
          (msg) => {
            if (msg.type === 'started') {
              setBatchProgress(p => ({ ...p, total: msg.total_runs, total_policies: msg.total_policies }));
            } else if (msg.type === 'progress') {
              setBatchProgress(p => ({
                ...p,
                completed: msg.completed,
                total: msg.total,
                current_policy: msg.current_policy,
                sim_idx: msg.sim_idx,
              }));
            } else if (msg.type === 'ground_truth') {
              setBatchProgress(p => ({ ...p, phase: 'ground_truth' }));
            }
          },
        );
        setBatchResult(result);
      } catch (e) {
        setError(String(e));
        throw e;
      } finally {
        setBatchProgress(null);
      }
      return;
    }

    // Single-policy mode: existing flow
    const { session_id } = await createSession({ policy, session_seed, sim_config, session_config });
    const effectiveBudget = budget ?? 10;
    const [post, kg] = await Promise.all([
      getPosterior(session_id),
      getKGComparison(session_id, 0.01, 50, effectiveBudget),
    ]);
    setSession({ id: session_id, policy, seed: session_seed, budget: budget ?? null });
    setPosterior(post);
    setKgComparison(kg);
    setBestCStar(post.best_c_star);
    setHistory([]);
    setNSteps(0);
    setLastResult(null);
    setReveal(null);
  }, []);

  // ── Human: evaluate at a chosen C* ───────────────────────────────────────

  const handleEvaluate = useCallback(async (cStar) => {
    if (!session || loading) return;
    setLoading(true);
    setError(null);
    try {
      const result = await evaluateC(session.id, cStar);
      const [post, kg] = await Promise.all([
        getPosterior(session.id),
        getKGComparison(session.id, 0.01, 50, session.budget ?? 10),
      ]);
      applyResult(result, post, kg);
      if (session.budget && result.n_steps >= session.budget) {
        await fetchReveal(session.id);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [session, loading, applyResult, fetchReveal]);

  // ── Automated: single step ────────────────────────────────────────────────

  const doStep = useCallback(async (sid, budget) => {
    const result = await runStep(sid);
    const [post, kg] = await Promise.all([
      getPosterior(sid),
      getKGComparison(sid, 0.01, 50, budget ?? 10),
    ]);
    applyResult(result, post, kg);
  }, [applyResult]);

  const handleStep = useCallback(async () => {
    if (!session || loading) return;
    setLoading(true);
    setError(null);
    try { await doStep(session.id, session.budget); }
    catch (e) { setError(String(e)); }
    finally   { setLoading(false); }
  }, [session, loading, doStep]);

  // ── Automated: auto-run N steps ───────────────────────────────────────────

  const handleAutoRun = useCallback(async (n) => {
    if (!session || loading) return;
    stopRef.current = false;
    setLoading(true);
    setError(null);
    try {
      for (let i = 0; i < n; i++) {
        if (stopRef.current) break;
        setAutoCount(n - i);
        await doStep(session.id, session.budget);
        await new Promise(r => setTimeout(r, 80));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      setAutoCount(0);
    }
  }, [session, loading, doStep]);

  // ── Reveal (automated mode manual trigger) ────────────────────────────────

  const handleReveal = useCallback(async () => {
    if (!session || revealLoading) return;
    await fetchReveal(session.id);
  }, [session, revealLoading, fetchReveal]);

  // ── New session ───────────────────────────────────────────────────────────

  const handleNew = useCallback(async () => {
    stopRef.current = true;
    if (session) await deleteSession(session.id).catch(() => {});
    setSession(null);
    setPosterior(null);
    setKgComparison(null);
    setHistory([]);
    setNSteps(0);
    setBestCStar(null);
    setLastResult(null);
    setReveal(null);
    setError(null);
    setBatchResult(null);
  }, [session]);

  // ── Render ────────────────────────────────────────────────────────────────

  if (batchResult) {
    return <BatchResults batch={batchResult} onReset={handleNew} />;
  }

  if (batchProgress) {
    const { completed, total, current_policy, sim_idx, family, phase } = batchProgress;
    const pct = total ? (100 * completed / total) : 0;
    const label = phase === 'ground_truth'
      ? 'Evaluating ground truth…'
      : current_policy
        ? `Sim ${sim_idx} · ${current_policy}`
        : 'Starting…';
    return (
      <div className="app">
        <div className="header">
          <h1>Running {family} batch</h1>
          <span className="badge">{completed} / {total ?? '?'}</span>
        </div>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>{label}</span>
            <span style={{ fontSize: 13, color: '#64748b', fontFamily: 'monospace' }}>
              {pct.toFixed(0)}%
            </span>
          </div>
          <div style={{ height: 10, background: '#e2e8f0', borderRadius: 9999, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${pct}%`,
              background: '#16a34a',
              borderRadius: 9999,
              transition: 'width 0.15s linear',
            }} />
          </div>
          <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 10 }}>
            Streaming progress as each policy-run completes. Keep this tab open;
            closing it cancels the batch on the server after the current run finishes.
          </p>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="app">
        <SessionForm onCreate={handleCreate} error={error} />
      </div>
    );
  }

  const { policy, seed, budget } = session;
  const isHuman = policy === 'human';

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <h1>Learning While Doing</h1>
        <span className={`badge badge-${policy}`}>{POLICY_LABEL[policy]}</span>
        <div className="session-meta">
          seed {seed}
          <button className="btn btn-ghost" onClick={handleNew}>← New session</button>
        </div>
      </div>

      {/* Stats */}
      <div className="card">
        <div style={{ display: 'flex', gap: 32, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="stats">
            <div className="stat">
              <span className="stat-label">Steps</span>
              <span className="stat-value">{nSteps}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Best C*</span>
              <span className="stat-value">
                {bestCStar != null ? bestCStar.toFixed(4) : '—'}
              </span>
            </div>
          </div>

          {/* Automated controls */}
          {!isHuman && (
            <div className="controls" style={{ marginLeft: 'auto' }}>
              {loading && autoCount > 0 && (
                <span style={{ fontSize: 13, color: '#64748b' }}>
                  {autoCount} step{autoCount !== 1 ? 's' : ''} left…
                </span>
              )}
              {loading && autoCount > 0 && (
                <button className="btn btn-outline" onClick={() => { stopRef.current = true; }}>
                  Stop
                </button>
              )}
              <button className="btn btn-outline" onClick={handleStep} disabled={loading}>
                Run step
              </button>
              <button className="btn btn-primary" onClick={() => handleAutoRun(10)} disabled={loading}>
                Auto-run 10
              </button>
              <button className="btn btn-primary" onClick={() => handleAutoRun(25)} disabled={loading}
                style={{ opacity: 0.85 }}>
                Auto-run 25
              </button>
              {nSteps > 0 && !reveal && (
                <button className="btn btn-outline" onClick={handleReveal}
                  disabled={loading || revealLoading}>
                  {revealLoading ? 'Computing…' : 'Reveal truth'}
                </button>
              )}
            </div>
          )}
        </div>

        {error && (
          <p style={{ color: '#dc2626', fontSize: 13, marginTop: 12 }}>{error}</p>
        )}
      </div>

      {/* Budget bar (human only) */}
      {isHuman && (
        <div className="card">
          <HumanControls budget={budget ?? 10} used={nSteps} />
        </div>
      )}

      {/* KG chart — placed above the posterior. For human mode, the aligned
          C* slider + Run button live directly below the plot so the user can
          eye-ball the KG curves and pick x visually. */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
            KG(x) — 1% grid
          </span>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            correlated (analytic vs MC) vs independent beliefs
          </span>
        </div>
        <p style={{ fontSize: 11, color: '#94a3b8', margin: '2px 0 8px 0' }}>
          Same underlying GP posterior; three ways of scoring each candidate C*.
          Analytic and MC should agree closely (any gap is MC noise).
          Independent ignores the covariance cascade to nearby points.
        </p>
        <KGChart kg={kgComparison} />
        {isHuman && (
          <CStarSlider
            cStar={cStar}
            setCStar={setCStar}
            onRun={() => handleEvaluate(cStar)}
            loading={loading}
            exhausted={budget != null && nSteps >= budget}
          />
        )}
      </div>

      {/* Posterior chart */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>GP posterior</span>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            {nSteps} observation{nSteps !== 1 ? 's' : ''}
          </span>
        </div>
        <PosteriorChart posterior={posterior} history={history} policy={policy} />
      </div>

      {/* Last simulation result — cash chart + jump log */}
      {lastResult && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
              Cash position — C* = {lastResult.c_star.toFixed(3)}
            </span>
            <span style={{ fontSize: 12, color: '#64748b' }}>
              Opp. cost {fmt(lastResult.opportunity_cost)} · Shortfall {fmt(lastResult.shortfall_cost)} · Total {fmt(lastResult.total_cost)}
            </span>
          </div>
          <CashChart result={lastResult} />
          <JumpLog events={lastResult.event_log} initialAum={lastResult.initial_aum} />
        </div>
      )}

      {/* History */}
      <div className="card">
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 12 }}>
          Observation history
        </div>
        <HistoryTable history={history} />
      </div>

      {/* End-of-session reveal */}
      {revealLoading && !reveal && (
        <div className="card" style={{ textAlign: 'center', color: '#64748b', fontSize: 13, padding: 32 }}>
          Computing true cost curve — running {30 * 12} simulations…
        </div>
      )}
      {reveal && <RevealPanel reveal={reveal} />}
    </div>
  );
}

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}
