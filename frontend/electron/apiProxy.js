'use strict';

/**
 * The only place the control-plane bearer token exists in the desktop app.
 *
 * The renderer never receives the token. It asks the main process to perform a
 * request (`api:request`), the main process attaches `Authorization`, and only
 * the status code and response body travel back over IPC. Anything that leaks
 * the token through a return value would defeat the point, so nothing here
 * returns it.
 *
 * Everything a renderer can influence is validated here rather than in the
 * preload script: the preload runs in a sandboxed renderer and its checks are
 * advisory at best once renderer JavaScript is compromised.
 */

const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { fileURLToPath } = require('node:url');

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;
const BACKEND_ORIGIN = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

/** Generous enough for a cold GPT-2 load, short enough to fail visibly. */
const REQUEST_TIMEOUT_MS = 180000;

/** Bodies are research payloads; a few MB is already far past normal use. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ALLOWED_METHODS = new Set(['GET', 'POST', 'PUT', 'PATCH', 'DELETE']);

/**
 * Only the control plane is proxied.
 *
 * The pattern is deliberately narrow: no scheme, no host, no `..`, no query
 * string smuggling (the query is a separate field), no traversal back to the
 * filesystem. `path.join` is never applied to it.
 */
const ALLOWED_PATH = /^\/(?:api|health)(?:\/[A-Za-z0-9._~-]+)*\/?$/;

/** Headers a renderer may set. `Authorization` is not one of them. */
const ALLOWED_HEADERS = new Set([
  'accept',
  'content-type',
  'x-request-id',
  'x-mech-client',
]);

/**
 * Read the token the backend accepts.
 *
 * Order: explicit operator override, then the token file the backend writes.
 * The path list mirrors the backend's `_token_file()` for the packaged layout
 * (`resources/backend/storage/.mech_api_token`) and the dev checkout.
 */
function resolveApiToken() {
  if (process.env.MECH_API_TOKEN) return process.env.MECH_API_TOKEN.trim();

  const repoRoot = path.join(__dirname, '..', '..');
  const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
  const candidates = [
    path.join(repoRoot, 'backend', 'storage', '.mech_api_token'),
    path.join(resourcesPath, 'backend', 'storage', '.mech_api_token'),
    path.join(process.cwd(), 'backend', 'storage', '.mech_api_token'),
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) {
        const value = fs.readFileSync(candidate, 'utf8').trim();
        if (value) return value;
      }
    } catch {
      /* unreadable candidate: try the next one */
    }
  }
  return '';
}

/**
 * Is this renderer URL one we are willing to serve API requests for?
 *
 * Dev: the Vite dev-server origin. Packaged: files under `dist/`. Anything
 * else -- a remote page, an about:blank frame, a data: URL -- is refused.
 */
function isTrustedRendererUrl(rawUrl) {
  if (!rawUrl) return false;
  let target;
  try {
    target = new URL(rawUrl);
  } catch {
    return false;
  }

  if (target.protocol === 'file:') {
    const distRoot = path.join(__dirname, '..', 'dist') + path.sep;
    try {
      return fileURLToPath(target).startsWith(distRoot);
    } catch {
      return false;
    }
  }

  if (target.protocol === 'http:' || target.protocol === 'https:') {
    const devUrl = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
    try {
      return target.origin === new URL(devUrl).origin;
    } catch {
      return false;
    }
  }

  return false;
}

function normaliseHeaders(input) {
  const out = {};
  if (!input || typeof input !== 'object') return out;
  for (const [name, value] of Object.entries(input)) {
    const lower = String(name).toLowerCase();
    // Authorization is attached by this module and may never be supplied.
    if (lower === 'authorization' || lower === 'host' || lower === 'content-length') continue;
    if (!ALLOWED_HEADERS.has(lower)) continue;
    if (typeof value !== 'string') continue;
    out[lower] = value;
  }
  return out;
}

/**
 * Reject anything malformed before it can reach the network.
 *
 * Returns the sanitised request. Throws on a violation: a rejected request must
 * be visible to the caller, not silently rewritten.
 */
function validateApiRequest(request) {
  if (!request || typeof request !== 'object') {
    throw new Error('api request must be an object');
  }

  const method = String(request.method || 'GET').toUpperCase();
  if (!ALLOWED_METHODS.has(method)) {
    throw new Error(`api request method not allowed: ${method}`);
  }

  const apiPath = typeof request.path === 'string' ? request.path : '';
  if (!ALLOWED_PATH.test(apiPath)) {
    throw new Error(`api request path not allowed: ${JSON.stringify(apiPath)}`);
  }
  // Belt and braces: the regex already forbids these, but a future loosening of
  // the pattern must not silently enable traversal.
  if (apiPath.includes('..') || apiPath.includes('://')) {
    throw new Error(`api request path not allowed: ${JSON.stringify(apiPath)}`);
  }

  let body = request.body;
  if (body !== undefined && body !== null && typeof body !== 'object') {
    throw new Error('api request body must be a JSON object when present');
  }

  let query = '';
  if (request.query && typeof request.query === 'object') {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(request.query)) {
      if (value === undefined || value === null) continue;
      if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
        throw new Error(`api request query parameter ${key} must be scalar`);
      }
      params.set(key, String(value));
    }
    const encoded = params.toString();
    if (encoded) query = `?${encoded}`;
  }

  return { method, apiPath: apiPath + query, body, headers: normaliseHeaders(request.headers) };
}

/**
 * Perform one authenticated request against the loopback backend.
 *
 * Resolves `{ status, body }` and never throws for an HTTP error status: the
 * caller decides what a 4xx/5xx means. Transport failures reject.
 */
function apiRequest(request) {
  const { method, apiPath, body, headers } = validateApiRequest(request);

  const token = resolveApiToken();
  if (!token) {
    return Promise.reject(
      new Error('API token unavailable: the backend cannot be authenticated from this build')
    );
  }

  const payload = body === undefined || body === null
    ? undefined
    : Buffer.from(JSON.stringify(body), 'utf8');
  if (payload && payload.length > MAX_BODY_BYTES) {
    return Promise.reject(new Error(`api request body too large: ${payload.length} bytes`));
  }

  const requestHeaders = {
    accept: 'application/json',
    ...headers,
    authorization: `Bearer ${token}`,
  };
  if (payload) {
    requestHeaders['content-type'] = headers['content-type'] || 'application/json';
    requestHeaders['content-length'] = String(payload.length);
  }

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: BACKEND_HOST,
        port: BACKEND_PORT,
        path: apiPath,
        method,
        headers: requestHeaders,
      },
      (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8');
          let parsed = null;
          if (text) {
            try {
              parsed = JSON.parse(text);
            } catch {
              parsed = text;
            }
          }
          resolve({ status: res.statusCode || 0, body: parsed });
        });
        res.on('error', reject);
      }
    );

    req.setTimeout(REQUEST_TIMEOUT_MS, () => {
      req.destroy(new Error(`backend request timed out after ${REQUEST_TIMEOUT_MS} ms`));
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

module.exports = {
  BACKEND_ORIGIN,
  MAX_BODY_BYTES,
  REQUEST_TIMEOUT_MS,
  apiRequest,
  isTrustedRendererUrl,
  resolveApiToken,
  validateApiRequest,
};
