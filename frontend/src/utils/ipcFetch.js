'use strict';

// Renderer transport shim: no local HTTP server exists in the desktop app.
// Any window.fetch() to http://localhost:8000/* is redirected through
// window.desktopApi.httpRequest() -> IPC 'mech:http' -> PythonBridge sidecar,
// which executes the FastAPI app in-process. Everything else uses native fetch.

const SIDECAR_HOSTS = new Set(['localhost:8000', '127.0.0.1:8000']);

function sidecarUrl(input) {
  try {
    const url = typeof input === 'string' ? new URL(input, window.location.href) : new URL(input.url);
    return SIDECAR_HOSTS.has(url.host) ? url : null;
  } catch {
    return null;
  }
}

async function shimFetch(input, init) {
  const url = sidecarUrl(input);
  if (!url || !window.desktopApi || typeof window.desktopApi.httpRequest !== 'function') {
    return window.__mechNativeFetch(input, init);
  }

  const requestInit = init || {};
  let body = requestInit.body;
  if (body != null && typeof body !== 'string') body = String(body);
  const headers = {};
  const srcHeaders = requestInit.headers || (typeof input !== 'string' && input.headers);
  if (srcHeaders) {
    if (typeof srcHeaders.forEach === 'function') {
      srcHeaders.forEach((value, key) => { headers[key] = value; });
    } else {
      Object.assign(headers, srcHeaders);
    }
  }

  const result = await window.desktopApi.httpRequest({
    method: requestInit.method || 'GET',
    path: url.pathname + url.search,
    headers,
    body: body ?? '',
  });

  return new Response(result.body || '', {
    status: result.status || 200,
    statusText: result.statusText || '',
    headers: result.headers || {},
  });
}

if (typeof window !== 'undefined' && !window.__mechIpcfetchInstalled) {
  window.__mechIpcfetchInstalled = true;
  window.__mechNativeFetch = window.fetch.bind(window);
  window.fetch = shimFetch;
}