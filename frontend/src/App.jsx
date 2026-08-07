import { useState, useCallback, useRef, useEffect } from 'react';
import { createSession, runStep, evaluateC, getPosterior, getPosterior2D, getKG2D, getReveal, getKGComparison, getKGvsM, setMStar, setZAlpha, setSigmaGreedy, setLengthScale, getObservationsEnriched, getFlowSample, runExperiment, runOneMore, resetSession, deleteSession } from './api';
import SessionForm from './components/SessionForm';
import PosteriorChart from './components/PosteriorChart';
import Belief3DChart from './components/Belief3DChart';
import KGChart from './components/KGChart';
import KGvsMChart from './components/KGvsMChart';
import ExperimentBar from './components/ExperimentBar';
import ImpparamSlider from './components/ImpparamSlider';
import HistoryTable from './components/HistoryTable';
import HumanControls from './components/HumanControls';
import CashChart from './components/CashChart';
import FlowSampleChart from './components/FlowSampleChart';
import JumpLog from './components/JumpLog';
import RevealPanel from './components/RevealPanel';
// BatchResults component intentionally not imported — the batch-mode
// flow was removed as part of the single-policy-with-tunable-parameter
// refactor (2026). Every policy now runs single-instance with its own
// θ^KGm* or θ^IE parameter chosen at session-create.
import './App.css';

// Small controlled number input that fires onCommit only on blur/Enter —
// prevents a request per keystroke while typing a new length_scale.
function LengthScaleInput({ value, onCommit }) {
  const [str, setStr] = useState(String(value));
  useEffect(() => { setStr(String(value)); }, [value]);
  const commit = () => {
    const v = Number(str);
    if (!Number.isFinite(v) || v <= 0) { setStr(String(value)); return; }
    if (Math.abs(v - value) < 1e-9) return;
    onCommit(v);
  };
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
                  fontSize: 12, color: '#475569' }}>
      <label style={{ fontWeight: 600 }}>Kernel bandwidth ℓ:</label>
      <input
        type="number"
        value={str}
        step="any"
        min="0"
        onChange={e => setStr(e.target.value)}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }}
        style={{ width: 80, padding: '3px 6px', border: '1px solid #cbd5e1',
                 borderRadius: 4, fontSize: 12 }}
      />
      <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>
        smaller ℓ = tighter local features · larger ℓ = smoother
      </span>
    </div>
  );
}

// 2-D variant: two independent inputs for per-dimension bandwidth (ARD).
// Accepts session.length_scale as either a scalar (broadcast to both dims
// when the session was created with the default) or a length-2 list.
// Commits an array [ℓ₁, ℓ₂] to the backend.
function LengthScaleInput2D({ value, onCommit }) {
  const initial = Array.isArray(value) ? value : [value, value];
  const [str1, setStr1] = useState(String(initial[0] ?? ''));
  const [str2, setStr2] = useState(String(initial[1] ?? ''));
  useEffect(() => {
    const v = Array.isArray(value) ? value : [value, value];
    setStr1(String(v[0] ?? ''));
    setStr2(String(v[1] ?? ''));
  }, [value]);
  const commit = () => {
    const v1 = Number(str1);
    const v2 = Number(str2);
    if (!Number.isFinite(v1) || v1 <= 0 || !Number.isFinite(v2) || v2 <= 0) {
      const orig = Array.isArray(value) ? value : [value, value];
      setStr1(String(orig[0]));
      setStr2(String(orig[1]));
      return;
    }
    const cur = Array.isArray(value) ? value : [value, value];
    if (Math.abs(v1 - cur[0]) < 1e-9 && Math.abs(v2 - cur[1]) < 1e-9) return;
    onCommit([v1, v2]);
  };
  const inputStyle = { width: 80, padding: '3px 6px', border: '1px solid #cbd5e1',
                       borderRadius: 4, fontSize: 12 };
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
                  fontSize: 12, color: '#475569', flexWrap: 'wrap' }}>
      <label style={{ fontWeight: 600 }}>Kernel bandwidth:</label>
      <label>ℓ₁:</label>
      <input type="number" value={str1} step="any" min="0" style={inputStyle}
             onChange={e => setStr1(e.target.value)}
             onBlur={commit}
             onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }} />
      <label>ℓ₂:</label>
      <input type="number" value={str2} step="any" min="0" style={inputStyle}
             onChange={e => setStr2(e.target.value)}
             onBlur={commit}
             onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }} />
      <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>
        per-dim ARD — smaller ℓ = tighter local features on that axis
      </span>
    </div>
  );
}

const POLICY_LABEL = {
  random:    'Random',
  ie:        'IE',
  kg:        'KG',
  kg_indep:  'KG indep',
  okg:       'OKG',
  okg_indep: 'OKG indep',
  human:     'Human',
};
// All KG variants share the green KG badge so the color still reads as "this
// is a KG-family policy" without needing four separate colors.
const POLICY_BADGE_KIND = {
  random:    'random',
  ie:        'ie',
  kg:        'kg',
  kg_indep:  'kg',
  okg:       'kg',
  okg_indep: 'kg',
  human:     'human',
};

// argmax over a 2-D grid encoded as a flat array in meshgrid(indexing="ij") order:
// values[i * len(axis2) + j] corresponds to (axis1[i], axis2[j]).
// Returns [axis1[i*], axis2[j*]] at the maximum, or null if inputs are missing.
function argmaxOnGrid(values, axis1, axis2) {
  if (!values?.length || !axis1?.length || !axis2?.length) return null;
  let best = -Infinity;
  let bestIdx = 0;
  for (let k = 0; k < values.length; k++) {
    const v = values[k];
    if (Number.isFinite(v) && v > best) { best = v; bestIdx = k; }
  }
  const j = bestIdx % axis2.length;
  const i = Math.floor(bestIdx / axis2.length);
  return [axis1[i], axis2[j]];
}

// Parse ?app=…&policy=…&auto=… once on mount. Landing page (CASTLE
// site) links here with these baked in — auto=1 means "skip the
// Advanced-parameters panel and go straight to the game". Anything
// invalid quietly falls back to defaults so a hand-typed URL can't
// break the app.
const VALID_APPS     = new Set(['cash_balance', 'cash_balance_2d']);
const VALID_POLICIES = new Set(['kg', 'kg_indep', 'okg', 'okg_indep', 'ie', 'random', 'human']);
function parseLaunchParams() {
  try {
    const p = new URLSearchParams(window.location.search);
    const app    = p.get('app');
    const policy = p.get('policy');
    const auto   = p.get('auto');
    return {
      initialAppName: VALID_APPS.has(app)         ? app    : 'cash_balance',
      initialPolicy:  VALID_POLICIES.has(policy)  ? policy : 'kg',
      autoSubmit:     auto === '1' || auto === 'true',
    };
  } catch {
    return { initialAppName: 'cash_balance', initialPolicy: 'kg', autoSubmit: false };
  }
}

export default function App() {
  // Read once — subsequent navigations inside the app should not
  // re-trigger auto-launch. Landing page → this app is a full page
  // load, so this runs fresh each time the user clicks Play.
  const launchParams = useRef(parseLaunchParams()).current;

  const [session,       setSession]       = useState(null);
  const [posterior,     setPosterior]     = useState(null);
  const [kgComparison,  setKgComparison]  = useState(null);
  const [kgVsM,         setKgVsM]         = useState(null);
  const [kgVsMSigmaEps, setKgVsMSigmaEps] = useState(null);   // user override for KG(m) σ_ε
  const [kgVsMMMax,     setKgVsMMMax]     = useState(50);      // user override for KG(m) x-axis extent
  const [kgVsMTheta,    setKgVsMTheta]    = useState(null);    // user override θ (or θ₁ for 2-D) — null = argmax(offline KG)
  const [kgVsMTheta2,   setKgVsMTheta2]   = useState(null);    // user override θ₂ (2-D only)
  const [kgVsMPending,  setKgVsMPending]  = useState(false);
  const [history,       setHistory]       = useState([]);
  const [enrichedRows,  setEnrichedRows]  = useState([]);
  // Score boxes on the ExperimentBar. `latestScore` = sum of rewards
  // from the last button press (Restart batch or single One more).
  // `cumulativeScore` = running total since the last Restart.
  const [latestScore,     setLatestScore]     = useState(null);
  const [cumulativeScore, setCumulativeScore] = useState(null);
  // Simulated days behind the cumulative score. Reset on Restart /
  // New session; incremented by n_days per iteration in One more.
  const [totalDays,       setTotalDays]       = useState(0);
  const [nSteps,        setNSteps]        = useState(0);
  const [bestImpparam,     setBestImpparam]     = useState(null);
  const [lastResult,    setLastResult]    = useState(null);
  const [reveal,        setReveal]        = useState(null);
  const [revealLoading, setRevealLoading] = useState(false);
  const [loading,       setLoading]       = useState(false);
  const [autoCount,     setAutoCount]     = useState(0);
  const [error,         setError]         = useState(null);
  // Render's free tier sleeps the backend after ~15 min idle; when it
  // wakes, in-memory sessions are gone and any next API call returns
  // 404 "session 'xxx' not found". Instead of showing that as a raw
  // error, auto-relaunch: reload the landing page with the current
  // app + auto=1 so a fresh session is created (using the user's
  // saved Advanced-params from localStorage). Small delay so the
  // error banner is visible for a beat before the redirect.
  useEffect(() => {
    if (!error) return;
    if (/\b404\b.*session.*not found/i.test(String(error))) {
      const app = session?.app_name ?? launchParams.initialAppName;
      const url = `https://warrenpowell.org/learning-while-doing/?app=${encodeURIComponent(app)}&auto=1`;
      setError('Session expired (backend was asleep) — starting a new one…');
      const t = setTimeout(() => { window.location.href = url; }, 800);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [error]);
  const [impparam,         setImpparam]         = useState(0.10);
  const [posterior2D,   setPosterior2D]   = useState(null);   // 2-D belief surface
  const [kg2D,          setKg2D]          = useState(null);   // 2-D KG surface
  const [flowSample,    setFlowSample]    = useState(null);   // deposits/redemptions sample path
  const stopRef = useRef(false);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const applyResult = useCallback((result, post, kg, kgm) => {
    setPosterior(post);
    setKgComparison(kg);
    if (kgm !== undefined) setKgVsM(kgm);
    setBestImpparam(result.best_impparam);
    setNSteps(result.n_steps);
    setHistory(prev => [...prev, [result.impparam, result.total_reward]]);
    setLastResult(result);
  }, []);

  // Fetch the enriched history (per-observation μ^n and KG at the
  // belief state BEFORE that step). Called after any event that adds an
  // observation OR changes the belief (length_scale refit) OR the m*
  // used in the KG calculation.
  const refreshEnriched = useCallback(async (sid) => {
    try {
      const resp = await getObservationsEnriched(sid);
      setEnrichedRows(resp.rows ?? []);
    } catch (e) {
      // Non-fatal — just leave the enriched columns off if the fetch fails.
      console.warn('observations_enriched fetch failed', e);
    }
  }, []);

  // User edits σ_ε or m_max on the KG(m) card — recompute the curve in
  // place. Overrides persist across subsequent steps; reset on new session.
  const handleKGvsMSigmaEps = useCallback(async (sigmaEps) => {
    if (!session) return;
    setKgVsMSigmaEps(sigmaEps);
    setKgVsMPending(true);
    try {
      const kgm = await getKGvsM(session.id, kgVsMMMax, sigmaEps, kgVsMTheta, kgVsMTheta2);
      setKgVsM(kgm);
    } catch (e) {
      setError(String(e));
    } finally {
      setKgVsMPending(false);
    }
  }, [session, kgVsMMMax, kgVsMTheta, kgVsMTheta2]);

  const handleKGvsMMMax = useCallback(async (newMMax) => {
    if (!session) return;
    setKgVsMMMax(newMMax);
    setKgVsMPending(true);
    try {
      const kgm = await getKGvsM(session.id, newMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2);
      setKgVsM(kgm);
    } catch (e) {
      setError(String(e));
    } finally {
      setKgVsMPending(false);
    }
  }, [session, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2]);

  // Policy m* (a.k.a. ρ^lkhd, "noise factor"): POST to
  // /sessions/{sid}/m_star and mirror the new value on session.m_star.
  // KG values everywhere the belief is displayed depend on m*, so
  // refresh the KG comparison (KGChart), the KG(m) card (KGvsMChart),
  // and the enriched history. Posterior does not depend on m*, no
  // refetch needed.
  const handleMStarChange = useCallback(async (newMStar) => {
    if (!session) return;
    try {
      const resp = await setMStar(session.id, newMStar);
      setSession(prev => prev ? { ...prev, m_star: resp.m_star } : prev);
      const refetches = [refreshEnriched(session.id)];
      if (session.dim === 1) {
        refetches.push(getKGComparison(session.id, 0.01, 50, 10).then(setKgComparison));
      }
      refetches.push(
        getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2).then(setKgVsM)
      );
      await Promise.all(refetches);
    } catch (e) {
      setError(String(e));
    }
  }, [session, refreshEnriched, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2]);

  // z_alpha (IE) — mid-session tunable via the ExperimentBar's
  // "policy parameter" slot. IE's argmin score = μ - z_alpha·σ, so the
  // KG(x) chart doesn't depend on z_alpha; no refetch needed there.
  const handleZAlphaChange = useCallback(async (newZAlpha) => {
    if (!session) return;
    try {
      const resp = await setZAlpha(session.id, newZAlpha);
      setSession(prev => prev ? { ...prev, z_alpha: resp.z_alpha } : prev);
    } catch (e) {
      setError(String(e));
    }
  }, [session]);

  // σ_greedy (RandomizedGreedy) — same story: doesn't affect displayed
  // KG or posterior; only the next proposal.
  const handleSigmaGreedyChange = useCallback(async (newSigmaGreedy) => {
    if (!session) return;
    try {
      const resp = await setSigmaGreedy(session.id, newSigmaGreedy);
      setSession(prev => prev ? { ...prev, sigma_greedy: resp.sigma_greedy } : prev);
    } catch (e) {
      setError(String(e));
    }
  }, [session]);

  // /experiment endpoint — one click, reset + K+1 iterations. Full
  // refresh of everything on the page afterwards. Belief config
  // (length_scale, m*) that the user set via mid-session controls is
  // preserved by the backend's reset (it rebuilds from the current
  // config, not the original one). Mirrors the policy the user chose
  // in the bar back onto session state so subsequent chart labels /
  // KG(m) card reflect it.
  // Shared refresh after any experiment endpoint returns — mirrors the
  // response into local state and re-fetches every view that depends on
  // the belief.
  const applyExperimentResponse = useCallback(async (resp, spec) => {
    // Mirror the new policy + parameter on the session record so
    // downstream UI (badge, KG(m) card sessionMStar, etc.) sees it.
    setSession(prev => prev ? {
      ...prev,
      policy: spec.policy ?? prev.policy,
      m_star: spec.m_star != null ? spec.m_star : prev.m_star,
    } : prev);
    setHistory(resp.history);
    setNSteps(resp.n_steps);
    setBestImpparam(resp.best_impparam);
    setLastResult(null);
    setReveal(null);
    const refetches = [];
    if (session.dim === 1) {
      refetches.push(getPosterior(session.id).then(setPosterior));
      refetches.push(getKGComparison(session.id, 0.01, 50, session.budget ?? 10).then(setKgComparison));
      refetches.push(getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2).then(setKgVsM));
    } else {
      refetches.push(getPosterior2D(session.id, 30).then(setPosterior2D));
      refetches.push(getKG2D(session.id, 20).then(setKg2D));
      refetches.push(getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2).then(setKgVsM));
    }
    refetches.push(refreshEnriched(session.id));
    await Promise.all(refetches);
  }, [session, refreshEnriched, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2]);

  // Refetch the deposits/redemptions sample with a specific θ so the
  // cash-balance line on that chart reflects whatever the user just
  // ran with. Backend default (0.10) is used when theta is null (fresh
  // session before any user action).
  const refreshFlowSample = useCallback(async (theta = null, theta2 = null) => {
    if (!session) return;
    const horizon = session.flow_horizon ?? 200;
    try {
      const flow = await getFlowSample(session.id, horizon, theta, theta2);
      setFlowSample(flow);
    } catch (e) { /* non-fatal — chart just keeps prior view */ }
  }, [session]);

  const handleRunExperiment = useCallback(async (spec) => {
    if (!session) return;
    setLoading(true); setError(null);
    try {
      const resp = await runExperiment(session.id, spec);
      await applyExperimentResponse(resp, spec);
      // Restart wipes prior state, so latest = cumulative = sum of the
      // freshly-generated batch's rewards.
      const rows = resp.history ?? [];
      const batchTotal = rows.reduce((s, row) => s + Number(row[1] ?? 0), 0);
      setLatestScore(batchTotal);
      setCumulativeScore(batchTotal);
      // Total simulated days = iterations × n_days. Restart resets.
      setTotalDays(rows.length * Number(spec.n_days ?? 0));
      // Update the cash-balance sample-path chart to use the θ the
      // user just chose (spec.theta_init). Both 1-D and 2-D supported.
      const t = spec.theta_init;
      if (Array.isArray(t) && t.length >= 2) {
        await refreshFlowSample(t[0], t[1]);
      } else if (t != null) {
        await refreshFlowSample(Number(t));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [session, applyExperimentResponse, refreshFlowSample]);

  // Restart: reset the backend session to initial conditions (empty
  // belief, empty history, re-seeded RNG) and zero all UI counters —
  // do NOT run any iteration. The dual-mode button flips back to
  // "Run" so the user can hit it to actually start a new experiment.
  const handleRestart = useCallback(async () => {
    if (!session) return;
    setLoading(true); setError(null);
    try {
      await resetSession(session.id);
      // Wipe UI state that describes an in-progress experiment.
      setHistory([]);
      setEnrichedRows([]);
      setNSteps(0);
      setLastResult(null);
      setLatestScore(null);
      setCumulativeScore(null);
      setTotalDays(0);
      setReveal(null);
      // Refresh the belief-derived views so they show the (unchanged)
      // prior instead of the last experiment's posterior.
      if (session.dim === 1) {
        const [post, kg, kgm] = await Promise.all([
          getPosterior(session.id),
          getKGComparison(session.id, 0.01, 50, 10),
          getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2),
        ]);
        setPosterior(post);
        setKgComparison(kg);
        setKgVsM(kgm);
        setBestImpparam(post.best_impparam);
      } else {
        const [p2, kg2, kgm] = await Promise.all([
          getPosterior2D(session.id, 30),
          getKG2D(session.id, 20),
          getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2),
        ]);
        setPosterior2D(p2);
        setKg2D(kg2);
        setKgVsM(kgm);
        setBestImpparam(p2.best_impparam);
      }
      // Restart also resets the ExperimentBar's Starting-point box to
      // its pre-fill (via its lastTheta=null useEffect); mirror that
      // on the cash-balance chart using the same initial θ the user
      // set in Advanced Parameters.
      const init = session.initial_theta;
      if (session.dim === 2) {
        const t1 = Array.isArray(init) ? Number(init[0]) : 0.1;
        const t2 = Array.isArray(init) ? Number(init[1]) : 0.1;
        await refreshFlowSample(t1, t2);
      } else {
        await refreshFlowSample(Number(init ?? 0.1));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [session, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2, refreshFlowSample]);

  const handleOneMore = useCallback(async (spec) => {
    if (!session) return;
    // Capture how many steps existed BEFORE the call so we can slice
    // the new-only rows out of the response afterwards.
    const priorNSteps = nSteps;
    setLoading(true); setError(null);
    try {
      const resp = await runOneMore(session.id, spec);
      await applyExperimentResponse(resp, spec);
      const newRows = (resp.history ?? []).slice(priorNSteps);
      const batchTotal = newRows.reduce((s, row) => s + Number(row[1] ?? 0), 0);
      setLatestScore(batchTotal);
      setCumulativeScore(prev => (prev ?? 0) + batchTotal);
      setTotalDays(prev => prev + newRows.length * Number(spec.n_days ?? 0));
      // Cash-balance chart follows the most recent θ (policy-picked here).
      const lastRow = (resp.history ?? []).at(-1);
      if (lastRow) {
        const t = lastRow[0];
        if (Array.isArray(t) && t.length >= 2) {
          await refreshFlowSample(t[0], t[1]);
        } else if (t != null) {
          await refreshFlowSample(Number(t));
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [session, nSteps, applyExperimentResponse, refreshFlowSample]);

  // GP length_scale (bandwidth) editor: POST refits the belief with the
  // new value replaying the session's history through it, then refetch
  // everything that depends on the posterior. Kept 1-D-only for now per
  // Warren's request (2-D adds a second-dim input for later).
  const handleLengthScaleChange = useCallback(async (newLS) => {
    if (!session) return;
    try {
      const resp = await setLengthScale(session.id, newLS);
      setSession(prev => prev ? { ...prev, length_scale: resp.length_scale } : prev);
      // Refetch every view that depends on the posterior.
      if (session.dim === 1) {
        const [post, kg, kgm] = await Promise.all([
          getPosterior(session.id),
          getKGComparison(session.id, 0.01, 50, session.budget ?? 10),
          getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2),
        ]);
        setPosterior(post);
        setKgComparison(kg);
        setKgVsM(kgm);
        setBestImpparam(post.best_impparam);
      } else {
        const [p2, kg2, kgm] = await Promise.all([
          getPosterior2D(session.id, 30),
          getKG2D(session.id, 20),
          getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2),
        ]);
        setPosterior2D(p2);
        setKg2D(kg2);
        setKgVsM(kgm);
        setBestImpparam(p2.best_impparam);
      }
      // μ and KG in the enriched history depend on the belief (which
      // just got refit) — always refresh regardless of dim.
      refreshEnriched(session.id);
    } catch (e) {
      setError(String(e));
    }
  }, [session, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2, refreshEnriched]);

  const handleKGvsMTheta = useCallback(async (newTheta, newTheta2 = null) => {
    if (!session) return;
    setKgVsMTheta(newTheta);      // null → backend picks argmax(offline KG)
    setKgVsMTheta2(newTheta2);    // ignored by 1-D endpoint; needed for 2-D
    setKgVsMPending(true);
    try {
      const kgm = await getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, newTheta, newTheta2);
      setKgVsM(kgm);
    } catch (e) {
      setError(String(e));
    } finally {
      setKgVsMPending(false);
    }
  }, [session, kgVsMMMax, kgVsMSigmaEps]);

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

  const handleCreate = useCallback(async ({
    app_name, policy, session_seed,
    sim_config, belief_config, acq_config, session_config,
    budget, m_star, report_level, flow_horizon, initial_theta,
  }) => {
    setError(null);
    // Single-policy mode is the only mode now; batch mode was removed.
    const created = await createSession({
      app_name: app_name ?? 'cash_balance',
      policy, session_seed,
      sim_config, belief_config, session_config,
      // Pass acq_config through when non-empty (IE sends z_alpha).
      ...(acq_config && Object.keys(acq_config).length ? { acq_config } : {}),
      // KG-family m* (θ^KGm*): applies to KG variants; harmless for others.
      ...(m_star != null ? { m_star } : {}),
    });
    const session_id = created.session_id;
    const dim = created.dim ?? 1;
    const effectiveBudget = budget ?? 10;

    setSession({
      id: session_id,
      app_name: created.app_name,
      dim,
      minimize: created.minimize,
      policy,
      seed: session_seed,
      budget: budget ?? null,
      m_star: created.m_star ?? 1,
      flow_horizon: Math.max(1, Math.min(5000, Math.round(Number(flow_horizon) || 200))),
      // z_alpha / sigma_greedy default to 0 unless the SessionForm
      // sent them in acq_config. Mirroring them on session state lets
      // the ExperimentBar's policy-parameter field show their current
      // values without an extra round-trip.
      z_alpha:      acq_config?.z_alpha      ?? 0,
      sigma_greedy: acq_config?.sigma_greedy ?? 0,
      // Whatever length_scale the user set in Advanced Parameters at
      // create time — the belief_config payload carries it.
      length_scale: belief_config?.length_scale ?? 0.04,
      // Frontend-only setting — controls which diagnostic panels render.
      report_level: report_level ?? 'basic',
      // Starting θ from Advanced Parameters. Pre-fills the
      // ExperimentBar's Starting-point box and seeds the initial
      // cash-balance chart. Scalar for 1-D, 2-vector for 2-D.
      initial_theta: initial_theta ?? (dim === 2 ? [0.1, 0.1] : 0.1),
    });
    setHistory([]);
    setEnrichedRows([]);
    setNSteps(0);
    setLastResult(null);
    setReveal(null);
    setLatestScore(null);
    setCumulativeScore(null);
    setTotalDays(0);
    setPosterior(null);
    setKgComparison(null);
    setKgVsM(null);
    setKgVsMSigmaEps(null);
    setKgVsMMMax(50);
    setKgVsMTheta(null);
    setKgVsMTheta2(null);
    setPosterior2D(null);
    setKg2D(null);
    setFlowSample(null);
    const horizon = Math.max(1, Math.min(5000, Math.round(Number(flow_horizon) || 200)));
    // Initial cash-line θ — same value that pre-fills the ExperimentBar's
    // Starting-point box. Falls back to 0.1 if nothing was supplied.
    const initTheta1 = Array.isArray(initial_theta) ? Number(initial_theta[0]) : Number(initial_theta ?? 0.1);
    const initTheta2 = Array.isArray(initial_theta) ? Number(initial_theta[1]) : null;

    if (dim === 1) {
      const [post, kg, kgm, flow] = await Promise.all([
        getPosterior(session_id),
        getKGComparison(session_id, 0.01, 50, effectiveBudget),
        getKGvsM(session_id, 50, null),   // fresh session — belief's σ_ε, default m_max
        getFlowSample(session_id, horizon, initTheta1),
      ]);
      setPosterior(post);
      setKgComparison(kg);
      setKgVsM(kgm);
      setBestImpparam(post.best_impparam);
      setFlowSample(flow);
    } else {
      // 2-D: fetch belief surface + KG surface + KG(m) card + flow sample in parallel.
      const [p2, kg2, kgm, flow] = await Promise.all([
        getPosterior2D(session_id, 30),
        getKG2D(session_id, 20),
        getKGvsM(session_id, 50, null, null, null),
        getFlowSample(session_id, horizon, initTheta1, initTheta2),
      ]);
      setPosterior2D(p2);
      setKg2D(kg2);
      setKgVsM(kgm);
      setBestImpparam(p2.best_impparam);
      setFlowSample(flow);
    }
  }, []);

  // ── Manual (human) policy: evaluate at a chosen θ ───────────────────────
  //
  // Used by both the ImpparamSlider (legacy human control) and by the
  // ExperimentBar's Run button when Manual is selected — one iteration
  // at the user-typed θ, no reset. Score boxes, totalDays, and the
  // cash-balance chart all follow.

  const handleEvaluate = useCallback(async (impparam, n_days = null) => {
    if (!session || loading) return;
    setLoading(true);
    setError(null);
    try {
      const result = await evaluateC(session.id, impparam, n_days);
      const [post, kg, kgm] = await Promise.all([
        getPosterior(session.id),
        getKGComparison(session.id, 0.01, 50, session.budget ?? 10),
        getKGvsM(session.id, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta),
      ]);
      applyResult(result, post, kg, kgm);
      refreshEnriched(session.id);
      // Score bookkeeping — Manual uses this instead of Run/Repeat's
      // batch-total accounting. Each click adds one iteration's reward.
      const reward = Number(result.total_reward ?? 0);
      const days   = Number(n_days ?? result.n_days ?? 0);
      setLatestScore(reward);
      setCumulativeScore(prev => (prev ?? 0) + reward);
      setTotalDays(prev => prev + days);
      // Cash chart tracks the θ the user just tested.
      if (Array.isArray(impparam) && impparam.length >= 2) {
        await refreshFlowSample(impparam[0], impparam[1]);
      } else if (impparam != null) {
        await refreshFlowSample(Number(impparam));
      }
      if (session.budget && result.n_steps >= session.budget) {
        await fetchReveal(session.id);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [session, loading, applyResult, fetchReveal, refreshEnriched, refreshFlowSample, kgVsMSigmaEps, kgVsMMMax, kgVsMTheta, kgVsMTheta2]);

  // ── Automated: single step ────────────────────────────────────────────────

  const doStep = useCallback(async (sid, budget, dim) => {
    const result = await runStep(sid);
    if (dim === 1) {
      const [post, kg, kgm] = await Promise.all([
        getPosterior(sid),
        getKGComparison(sid, 0.01, 50, budget ?? 10),
        getKGvsM(sid, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta),
      ]);
      applyResult(result, post, kg, kgm);
    } else {
      const [p2, kg2, kgm] = await Promise.all([
        getPosterior2D(sid, 30),
        getKG2D(sid, 20),
        getKGvsM(sid, kgVsMMMax, kgVsMSigmaEps, kgVsMTheta, kgVsMTheta2),
      ]);
      setPosterior2D(p2);
      setKg2D(kg2);
      setKgVsM(kgm);
      setBestImpparam(result.best_impparam ?? p2.best_impparam);
      setNSteps(result.n_steps);
      setHistory(prev => [...prev, [result.impparam, result.total_reward]]);
      setLastResult(result);
    }
    // Always refresh enriched history — both dims benefit from μ/KG per row.
    refreshEnriched(sid);
  }, [applyResult, refreshEnriched, kgVsMSigmaEps, kgVsMMMax, kgVsMTheta, kgVsMTheta2]);

  const handleStep = useCallback(async () => {
    if (!session || loading) return;
    setLoading(true);
    setError(null);
    try { await doStep(session.id, session.budget, session.dim); }
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
        await doStep(session.id, session.budget, session.dim);
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

  // ── Return to game page ──────────────────────────────────────────────────

  // Clean up the backend session and navigate back to the CASTLE
  // landing page. If the game was opened in a new tab (via the
  // landing page's target="_blank" link), try to close the tab so
  // the user returns to the original one; browsers often block
  // scripted close for non-script-opened tabs, so fall back to a
  // same-tab navigation which always works.
  const handleReturnToLanding = useCallback(async () => {
    stopRef.current = true;
    if (session) {
      try { await deleteSession(session.id); } catch { /* best-effort */ }
    }
    const app = session?.app_name ?? launchParams.initialAppName;
    const url = `https://warrenpowell.org/learning-while-doing/?app=${encodeURIComponent(app)}`;
    try { window.close(); } catch (_) { /* no-op */ }
    // If close was blocked, we're still here — navigate.
    setTimeout(() => { window.location.href = url; }, 50);
  }, [session, launchParams]);

  // ── Render ────────────────────────────────────────────────────────────────

  if (!session) {
    // autoSubmit only applies to the very first render (fresh page
    // load from the landing page). After the user clicks "New session"
    // they should see the panel, so we consume the flag by tracking
    // how many times we've been rendered without a session.
    const autoSubmit = launchParams.autoSubmit && autoCount === 0;
    return (
      <div className="app">
        <SessionForm
          onCreate={handleCreate}
          error={error}
          initialAppName={launchParams.initialAppName}
          initialPolicy={launchParams.initialPolicy}
          autoSubmit={autoSubmit}
        />
      </div>
    );
  }

  const { policy, seed, budget } = session;
  const isHuman = policy === 'human';

  return (
    <div className="app app-wide">
      {/* Header */}
      <div className="header">
        <h1>Learning While Doing</h1>
        <span className={`badge badge-${POLICY_BADGE_KIND[policy] ?? policy}`}>
          {POLICY_LABEL[policy] ?? policy}
        </span>
        <div className="session-meta">
          seed {seed}
          <button className="btn btn-ghost" onClick={handleReturnToLanding}>← Return to game page</button>
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
              <span className="stat-label">Best θ</span>
              <span className="stat-value">
                {bestImpparam == null
                  ? '—'
                  : Array.isArray(bestImpparam)
                    ? `(${bestImpparam.map(v => v.toFixed(3)).join(', ')})`
                    : Number(bestImpparam).toFixed(4)}
              </span>
            </div>
          </div>

          {/* Reveal truth stays here — it's a session-scoped diagnostic,
              not an experiment control. Auto-run / Run-step buttons
              were replaced by the ExperimentBar below the Stats card. */}
          {session.dim === 1 && (
            <div className="controls" style={{ marginLeft: 'auto' }}>
              <button className="btn btn-outline" onClick={handleReveal}
                disabled={loading || revealLoading || reveal != null}
                title={reveal != null
                  ? 'Ground-truth reward curve already shown below'
                  : 'Plot the true underlying reward curve F(θ) to compare with your beliefs'}>
                {revealLoading ? 'Computing…' : reveal != null ? 'Truth revealed' : 'Reveal truth'}
              </button>
            </div>
          )}
        </div>

        {error && (
          <p style={{ color: '#dc2626', fontSize: 13, marginTop: 12 }}>{error}</p>
        )}
      </div>

      {/* Experiment control bar — the ONE way to run policies now.
          Each Run resets the session belief + history and executes
          K+1 iterations. See ExperimentBar for layout. */}
      <ExperimentBar
        dim={session.dim}
        defaultPolicy={session.policy}
        defaultMStar={session.m_star}
        defaultZAlpha={0}
        running={loading}
        onRun={handleRunExperiment}
        onOneMore={handleOneMore}
        onRestart={handleRestart}
        onManualEvaluate={handleEvaluate}
        initialTheta={session.initial_theta}
        canOneMore={nSteps > 0}
        latestScore={latestScore}
        cumulativeScore={cumulativeScore}
        totalDays={totalDays}
        lastTheta={history.length > 0 ? history[history.length - 1][0] : null}
        sessionMStar={session.m_star ?? 1}
        sessionZAlpha={session.z_alpha ?? 0}
        sessionSigmaGreedy={session.sigma_greedy ?? 0}
        onMStarChange={handleMStarChange}
        onZAlphaChange={handleZAlphaChange}
        onSigmaGreedyChange={handleSigmaGreedyChange}
      />

      {/* Budget bar (human only) — informational, tracks n_steps vs the
          user-picked budget. Kept as a courtesy since Human policy uses
          K=0 and the user cycles through Runs manually. */}
      {isHuman && (
        <div className="card">
          <HumanControls budget={budget ?? 10} used={nSteps} />
        </div>
      )}

      {/* Two-column layout. LEFT: KG chart on top with GP posterior
          directly under it, so both share the θ axis and the reader can
          eyeball how the sample-value peaks (KG) line up with the
          posterior structure. RIGHT: the reveal panel goes first when it
          exists (so hitting "Reveal truth" produces an immediately
          visible plot at the top-right), followed by the cash-position
          card and the observation history. */}
      <div className="chart-grid">
        {/* ── LEFT column ────────────────────────────────────────────── */}
        <div className="chart-col">
          {session.dim >= 2 && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                  Knowledge gradient surface — KG(θ)
                </span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>
                  info value at each candidate
                </span>
              </div>
              <p style={{ fontSize: 11, color: '#94a3b8', margin: '2px 0 8px 0' }}>
                Analytic correlated KG at every θ on a 20×20 probe grid.  The
                surface flattens (drops) as each observation reduces uncertainty
                in its neighborhood.  Red dots on the base plane mark past
                observations; green circle marks argmax KG — the θ this policy
                would try next.
              </p>
              <Belief3DChart data={kg2D && {
                               ...kg2D,
                               value: kg2D.kg,
                               best_impparam: argmaxOnGrid(kg2D.kg, kg2D.axis1, kg2D.axis2),
                             }}
                             valueLabel="KG(θ)"
                             colorScheme="jet"
                             obsMode="baseplane"
                             dollarZ={false}
                             bestLabel="next θ"
                             emptyMessage="Waiting for KG surface…" />
            </div>
          )}
          {session.dim >= 2 && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                  GP posterior surface — 3-D belief
                </span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>
                  {nSteps} observation{nSteps !== 1 ? 's' : ''}
                </span>
              </div>
              <p style={{ fontSize: 11, color: '#94a3b8', margin: '2px 0 8px 0' }}>
                Posterior mean of F(θ) as a 3-D surface over the 2-parameter box.
                Red dots are past observations at their realized (noisy) reward.
                Green circle on the base plane marks the current best θ.
              </p>
              <LengthScaleInput2D
                value={session.length_scale ?? 0.04}
                onCommit={handleLengthScaleChange}
              />
              <Belief3DChart data={posterior2D && { ...posterior2D, value: posterior2D.mean }}
                             valueLabel="Posterior mean reward"
                             colorScheme="viridis"
                             obsMode="atCost"
                             emptyMessage="Waiting for posterior…" />
            </div>
          )}

          {/* KG(m) card is available in 2-D too. Same semantics as 1-D:
              future-batch scenario at a caller-picked (θ₁, θ₂). Only
              rendered under Advanced reporting per Warren's request —
              basic reporting stays focused on the core pedagogical panels. */}
          {session.dim >= 2 && session.report_level === 'advanced' && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                  KG vs batch size m
                </span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>
                  at (θ₁, θ₂)* = argmax offline KG
                </span>
              </div>
              <p style={{ fontSize: 11, color: '#94a3b8', margin: '2px 0 8px 0' }}>
                Info value at the (θ₁, θ₂) point if we averaged m independent
                runs there (noise → σ/√m, precision → m·β). Uses the correlated
                GP posterior — the same one the KG policy consumes. σ_ε is a
                future-batch scenario; the session's belief is not refit.
              </p>
              <KGvsMChart
                data={kgVsM}
                onSigmaEpsChange={handleKGvsMSigmaEps}
                sigmaEpsPending={kgVsMPending}
                mMax={kgVsMMMax}
                onMMaxChange={handleKGvsMMMax}
                onThetaChange={handleKGvsMTheta}
                dim={2}
                sessionMStar={session.m_star ?? 1}
                onMStarChange={handleMStarChange}
              />
            </div>
          )}

          {session.dim === 1 && (
            <>
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                    KG(x) — 1% grid
                  </span>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    correlated (analytic vs MC) vs independent beliefs
                  </span>
                </div>
                <KGChart
                  kg={kgComparison}
                  sessionMStar={session.m_star ?? 1}
                  onMStarChange={handleMStarChange}
                />
                {isHuman && (
                  <ImpparamSlider
                    impparam={impparam}
                    setImpparam={setImpparam}
                    onRun={() => handleEvaluate(impparam)}
                    loading={loading}
                    exhausted={budget != null && nSteps >= budget}
                  />
                )}
              </div>

              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>GP posterior</span>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    {nSteps} observation{nSteps !== 1 ? 's' : ''}
                  </span>
                </div>
                <LengthScaleInput
                  value={session.length_scale ?? 0.04}
                  onCommit={handleLengthScaleChange}
                />
                <PosteriorChart posterior={posterior} history={history} policy={policy} />
              </div>

              {/* KG(θ*; m) — S-curve diagnostic. Only rendered under
                  Advanced reporting; basic keeps the layout focused on
                  the core pedagogical panels. */}
              {session.report_level === 'advanced' && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                    KG vs batch size m
                  </span>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    at θ* = argmax offline KG
                  </span>
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', margin: '2px 0 8px 0' }}>
                  Info value at θ if we averaged m independent runs there
                  (noise → σ/√m, precision → m·β). Uses the correlated GP
                  posterior — the same one the KG policy consumes. The σ_ε
                  input is a <b>future-batch scenario</b>: "what if the
                  hypothetical m runs had noise σ_ε?" — the session's belief
                  (posterior μ and V) is <em>not</em> refit, so Δ stays
                  substantial while σ̃(m=1) collapses at high σ_ε — that's
                  when the classical S emerges.
                </p>
                <KGvsMChart
                  data={kgVsM}
                  onSigmaEpsChange={handleKGvsMSigmaEps}
                  sigmaEpsPending={kgVsMPending}
                  mMax={kgVsMMMax}
                  onMMaxChange={handleKGvsMMMax}
                  onThetaChange={handleKGvsMTheta}
                  dim={1}
                  sessionMStar={session.m_star ?? 1}
                  onMStarChange={handleMStarChange}
                />
              </div>
              )}
            </>
          )}
        </div>

        {/* ── RIGHT column ───────────────────────────────────────────── */}
        <div className="chart-col">
          {/* Reveal panel first so it appears at the top-right when the
              user clicks "Reveal truth" — same width as other cards,
              impossible to miss. 1-D apps only. */}
          {session.dim === 1 && revealLoading && !reveal && (
            <div className="card" style={{ textAlign: 'center', color: '#64748b', fontSize: 13, padding: 32 }}>
              Computing true reward curve — running {30 * 12} simulations…
            </div>
          )}
          {session.dim === 1 && reveal && <RevealPanel reveal={reveal} />}

          {/* Deposits & redemptions sample path — exogenous flows the
              fund would see even without a policy choice. Horizon is
              set in Advanced parameters (default 200 days). */}
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                Deposits &amp; redemptions — sample path
              </span>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>
                {flowSample ? `H = ${flowSample.horizon} days` : ''}
              </span>
            </div>
            <FlowSampleChart sample={flowSample} />
          </div>

          {lastResult && session.dim === 1 && lastResult.cash_series && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
                  Cash position — θ = {Number(lastResult.impparam).toFixed(3)}
                </span>
                <span style={{ fontSize: 12, color: '#64748b' }}>
                  Market {fmt(lastResult.market_gain)} · Cash {fmt(lastResult.cash_gain)} · Shortfall −{fmt(lastResult.shortfall_penalty)} · Reward {fmt(lastResult.total_reward)}
                </span>
              </div>
              <CashChart result={lastResult} />
              <JumpLog events={lastResult.event_log} initialAum={lastResult.initial_aum} />
            </div>
          )}

          <div className="card">
            <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 12 }}>
              Observation history
            </div>
            <HistoryTable history={history} enrichedRows={enrichedRows} />
          </div>
        </div>
      </div>
    </div>
  );
}

function fmt(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}
