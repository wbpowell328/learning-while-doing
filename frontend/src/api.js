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
export const evaluateC   = (sid, c_star) => call(`/sessions/${sid}/evaluate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ c_star }),
});
export const getPosterior  = (sid) => call(`/sessions/${sid}/posterior`);
export const getReveal     = (sid) => call(`/sessions/${sid}/reveal`);
export const getKGComparison = (sid, spacing = 0.05, mcSamples = 500) =>
  call(`/sessions/${sid}/kg?spacing=${spacing}&mc_samples=${mcSamples}`);
export const deleteSession = (sid) => call(`/sessions/${sid}`, { method: 'DELETE' });
