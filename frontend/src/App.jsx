import { useState, useCallback, useRef } from 'react';
import { createSession, runStep, evaluateC, getPosterior, getReveal, deleteSession } from './api';
import SessionForm from './components/SessionForm';
import PosteriorChart from './components/PosteriorChart';
import HistoryTable from './components/HistoryTable';
import HumanControls from './components/HumanControls';
import CashChart from './components/CashChart';
import JumpLog from './components/JumpLog';
import RevealPanel from './components/RevealPanel';
import './App.css';

const POLICY_LABEL = { random: 'Random', ie: 'IE', kg: 'KG', human: 'Human' };

export default function App() {
  const [session,       setSession]       = useState(null);
  const [posterior,     setPosterior]     = useState(null);
  const [history,       setHistory]       = useState([]);
  const [nSteps,        setNSteps]        = useState(0);
  const [bestCStar,     setBestCStar]     = useState(null);
  const [lastResult,    setLastResult]    = useState(null);
  const [reveal,        setReveal]        = useState(null);
  const [revealLoading, setRevealLoading] = useState(false);
  const [loading,       setLoading]       = useState(false);
  const [autoCount,     setAutoCount]     = useState(0);
  const [error,         setError]         = useState(null);
  const stopRef = useRef(false);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const applyResult = useCallback((result, post) => {
    setPosterior(post);
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

  const handleCreate = useCallback(async ({ policy, session_seed, sim_config, session_config, budget }) => {
    setError(null);
    const { session_id } = await createSession({ policy, session_seed, sim_config, session_config });
    const post = await getPosterior(session_id);
    setSession({ id: session_id, policy, seed: session_seed, budget: budget ?? null });
    setPosterior(post);
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
      const post   = await getPosterior(session.id);
      applyResult(result, post);
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

  const doStep = useCallback(async (sid) => {
    const result = await runStep(sid);
    const post   = await getPosterior(sid);
    applyResult(result, post);
  }, [applyResult]);

  const handleStep = useCallback(async () => {
    if (!session || loading) return;
    setLoading(true);
    setError(null);
    try { await doStep(session.id); }
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
        await doStep(session.id);
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
    setHistory([]);
    setNSteps(0);
    setBestCStar(null);
    setLastResult(null);
    setReveal(null);
    setError(null);
  }, [session]);

  // ── Render ────────────────────────────────────────────────────────────────

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

      {/* Human controls */}
      {isHuman && (
        <div className="card">
          <HumanControls
            onEvaluate={handleEvaluate}
            loading={loading}
            budget={budget ?? 10}
            used={nSteps}
          />
        </div>
      )}

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
