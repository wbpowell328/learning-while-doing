const BASE = 'http://localhost:8000';

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
export const deleteSession = (sid) => call(`/sessions/${sid}`, { method: 'DELETE' });
