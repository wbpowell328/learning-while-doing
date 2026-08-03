// In production (single Render service) the API and UI share an origin, so
// BASE is empty and requests are same-origin. For local dev with the split
// Vite (5173) + FastAPI (8000) setup, set VITE_API_BASE=http://localhost:8000
// in frontend/.env.local (or use Vite's proxy option).
const BASE = import.meta.env.VITE_API_BASE ?? '';

async function call(path, options = {}) {
  const r = await fetch(`${BASE}${path}`, options);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  if (r.status === 204) return null;
  return r.json();
}

export const createSession = (body) =>
  call('/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const runStep     = (sid) => call(`/sessions/${sid}/step`, { method: 'POST' });
export const evaluateC   = (sid, impparam) => call(`/sessions/${sid}/evaluate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ impparam }),
});
export const getPosterior  = (sid) => call(`/sessions/${sid}/posterior`);
export const getPosterior2D = (sid, gridSize = 30) =>
  call(`/sessions/${sid}/posterior_2d?grid_size=${gridSize}`);
export const getKG2D = (sid, gridSize = 20) =>
  call(`/sessions/${sid}/kg_2d?grid_size=${gridSize}`);
export const getReveal     = (sid) => call(`/sessions/${sid}/reveal`);
export const getKGComparison = (sid, spacing = 0.05, mcSamples = 500, budget = 10) =>
  call(`/sessions/${sid}/kg?spacing=${spacing}&mc_samples=${mcSamples}&budget=${budget}`);
export const getKGvsM = (sid, mMax = 50, sigmaEps = null) => {
  const suffix = sigmaEps == null ? '' : `&sigma_eps=${sigmaEps}`;
  return call(`/sessions/${sid}/kg_vs_m?m_max=${mMax}${suffix}`);
};

// Streams the batch endpoint as NDJSON. Invokes onEvent(msg) for each
// {"started"|"progress"|"ground_truth"|"result"} message. Resolves with the
// final "result" payload (also delivered via onEvent).
//
// Throws an explicit error if the stream ends without a "result" event — the
// UI needs to know something went wrong rather than silently returning null.
export async function runBatch(body, onEvent) {
  const r = await fetch(`/sessions/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult = null;
  const seenTypes = new Set();

  const flushLines = (lines) => {
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch (err) {
        console.warn('[runBatch] failed to parse NDJSON line', {
          length: trimmed.length,
          head: trimmed.slice(0, 120),
          tail: trimmed.slice(-120),
          err: String(err),
        });
        continue;
      }
      seenTypes.add(msg.type);
      if (msg.type === 'result') finalResult = msg;
      onEvent?.(msg);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    flushLines(lines);
  }

  // Flush pending decoder bytes and process any final leftover in the buffer.
  buffer += decoder.decode();
  if (buffer.trim().length > 0) flushLines([buffer]);

  if (!finalResult) {
    throw new Error(
      `Batch stream ended without a "result" event. Received event types: [${[...seenTypes].join(', ') || 'none'}]. ` +
      `Check server logs — the connection may have been dropped mid-response.`
    );
  }
  return finalResult;
}
export const deleteSession = (sid) => call(`/sessions/${sid}`, { method: 'DELETE' });
