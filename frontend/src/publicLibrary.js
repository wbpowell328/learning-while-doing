// Public parameter-panel library for the cash-management game.
//
// Public panels are JSON files in the sda-website repo under
// assets/lwd-panels/, served read-only (with CORS) from warrenpowell.org.
// READING needs no auth — anyone can load a curated panel. PUBLISHING is
// admin-only (?admin=1): the admin pastes a fine-grained GitHub PAT once,
// it lives only in this browser's localStorage and is sent only to
// api.github.com, and each Publish / Rename / Delete is a real commit to
// the repo (GitHub Pages then rebuilds in ~90s). Same pattern as the
// decision-framing public examples.
//
// A public panel file is: { app, title, description, blob } where `blob`
// is exactly the private-panel blob SessionForm stores in localStorage
// (seed, mStar, zAlpha, sigmaGreedy, flowHorizon, reportLevel, adv, …),
// so loading a public panel is just importing it as a private panel.

const PUBLIC_BASE   = 'https://warrenpowell.org/assets/lwd-panels/';
const MANIFEST_PATH = 'assets/lwd-panels/index.json';
const FILES_DIR     = 'assets/lwd-panels';
const REPO_OWNER    = 'wbpowell328';
const REPO_NAME     = 'sda-website';
const TOKEN_KEY     = 'lwd_admin_gh_token_v1';
export const PAT_HELP_URL = 'https://github.com/settings/personal-access-tokens/new';

// ── Reading (public, no auth) ───────────────────────────────────────────────

export async function fetchPublicIndex() {
  try {
    const r = await fetch(PUBLIC_BASE + 'index.json', { cache: 'no-cache' });
    if (!r.ok) return [];                        // not deployed yet / 404 → empty
    const data = await r.json();
    return Array.isArray(data?.panels) ? data.panels : [];
  } catch (_) {
    return [];                                   // offline / CORS hiccup → empty
  }
}

export async function fetchPublicPanel(file) {
  const r = await fetch(PUBLIC_BASE + file, { cache: 'no-cache' });
  if (!r.ok) throw new Error('Could not load public panel "' + file + '" (' + r.status + ').');
  return r.json();
}

// ── Admin token (this browser only) ─────────────────────────────────────────

export function isAdminOn() {
  try { return new URL(window.location.href).searchParams.get('admin') === '1'; }
  catch (_) { return false; }
}
export function getAdminToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (_) { return ''; }
}
export function setAdminToken(tok) {
  try {
    if (tok) localStorage.setItem(TOKEN_KEY, tok);
    else localStorage.removeItem(TOKEN_KEY);
  } catch (_) { /* private mode / quota — ignore */ }
}

// ── GitHub Contents API helpers (admin, direct browser → api.github.com) ─────

function ghHeaders() {
  const tok = getAdminToken();
  if (!tok) throw new Error('No GitHub token set — click "Set / change token" first.');
  return {
    Authorization: 'Bearer ' + tok,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  };
}
function ghError(resp, bodyText) {
  if (resp.status === 401) return 'GitHub rejected the token (401) — it may be expired or wrong. Set a new one.';
  if (resp.status === 403) return 'GitHub returned 403 — the token lacks write access. Regenerate with Contents: Read and write on ' + REPO_NAME + '.';
  if (resp.status === 404) return 'GitHub returned 404 — file not found (deleted or renamed?).';
  if (resp.status === 409 || resp.status === 422) return 'GitHub returned ' + resp.status + ' — likely a stale SHA. Reload and retry. ' + bodyText.slice(0, 160);
  return 'GitHub request failed (' + resp.status + '): ' + bodyText.slice(0, 240);
}
// UTF-8-safe base64 (atob/btoa alone break on multibyte characters).
function b64EncodeUtf8(str) { return btoa(unescape(encodeURIComponent(str))); }
function b64DecodeUtf8(b64) { return decodeURIComponent(escape(atob(String(b64).replace(/\n/g, '')))); }

const CONTENTS = (path) =>
  'https://api.github.com/repos/' + REPO_OWNER + '/' + REPO_NAME + '/contents/' + path;

async function ghGetFile(path) {
  const resp = await fetch(CONTENTS(path), { headers: ghHeaders() });
  const text = await resp.text();
  if (!resp.ok) throw new Error(ghError(resp, text));
  const data = JSON.parse(text);
  return { sha: data.sha, content: b64DecodeUtf8(data.content || '') };
}
async function ghPutFile(path, content, message, sha) {
  const body = { message, content: b64EncodeUtf8(content) };
  if (sha) body.sha = sha;
  const resp = await fetch(CONTENTS(path), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...ghHeaders() },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  if (!resp.ok) throw new Error(ghError(resp, text));
  return JSON.parse(text);
}
async function ghDeleteFile(path, message, sha) {
  const resp = await fetch(CONTENTS(path), {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', ...ghHeaders() },
    body: JSON.stringify({ message, sha }),
  });
  const text = await resp.text();
  if (!resp.ok) throw new Error(ghError(resp, text));
  return JSON.parse(text);
}

export function slugify(text) {
  return String(text || '').toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60) || 'panel';
}

async function readManifest() {
  const file = await ghGetFile(MANIFEST_PATH);
  let obj;
  try { obj = JSON.parse(file.content || '{"panels":[]}'); }
  catch (_) { obj = { panels: [] }; }
  if (!Array.isArray(obj.panels)) obj.panels = [];
  return { sha: file.sha, obj };
}

// ── Admin operations (each = a couple of commits) ────────────────────────────

// Publish a panel blob into the public library: write the panel file, then
// add it to the manifest. Auto-uniquifies the slug on collision.
export async function publishPanel({ app, title, description, blob }) {
  const { sha, obj } = await readManifest();
  const existing = new Set(obj.panels.map(p => p.file));
  const base = slugify(app + '-' + title);
  let slug = base, n = 1;
  while (existing.has(slug + '.json')) { n++; slug = base + '-' + n; }
  const filename = slug + '.json';

  const fileObj = { app, title, description, blob };
  await ghPutFile(FILES_DIR + '/' + filename,
    JSON.stringify(fileObj, null, 2) + '\n',
    'lwd-panels: publish "' + title + '"');

  obj.panels.push({ file: filename, app, title, description });
  await ghPutFile(MANIFEST_PATH,
    JSON.stringify(obj, null, 2) + '\n',
    'lwd-panels: index "' + title + '"', sha);
  return filename;
}

export async function renamePublicPanel(file, newTitle, newDescription) {
  const f = await ghGetFile(FILES_DIR + '/' + file);
  let obj;
  try { obj = JSON.parse(f.content); } catch (_) { obj = {}; }
  obj.title = newTitle;
  obj.description = newDescription;
  await ghPutFile(FILES_DIR + '/' + file,
    JSON.stringify(obj, null, 2) + '\n',
    'lwd-panels: rename "' + newTitle + '"', f.sha);

  const { sha, obj: man } = await readManifest();
  man.panels = man.panels.map(p =>
    p.file === file ? { ...p, title: newTitle, description: newDescription } : p);
  await ghPutFile(MANIFEST_PATH,
    JSON.stringify(man, null, 2) + '\n',
    'lwd-panels: reindex "' + newTitle + '"', sha);
}

export async function deletePublicPanel(file) {
  const f = await ghGetFile(FILES_DIR + '/' + file);
  await ghDeleteFile(FILES_DIR + '/' + file, 'lwd-panels: delete ' + file, f.sha);

  const { sha, obj } = await readManifest();
  obj.panels = obj.panels.filter(p => p.file !== file);
  await ghPutFile(MANIFEST_PATH,
    JSON.stringify(obj, null, 2) + '\n',
    'lwd-panels: de-index ' + file, sha);
}
