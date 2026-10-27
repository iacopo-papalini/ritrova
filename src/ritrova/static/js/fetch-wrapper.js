/* ===================================================================
   fetch-wrapper: global fetch interceptor + _safeToast helper
   -------------------------------------------------------------------
   Auto-toasts on non-2xx responses and network errors so mutation
   actions never fail silently (BUG-20). Opt-out by setting
   { skipErrorToast: true } on the fetch init object.

   409 Conflict is reserved for "needs_confirm" flows (see BUG-19 —
   cross-species assignment), where the UI deliberately handles the
   response. We skip the auto-toast for 409 so it doesn't fire
   alongside a confirm dialog.
   =================================================================== */

const CSRF_COOKIE = 'ritrova_csrf';
const CSRF_HEADER = 'X-CSRF-Token';
const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

/* The server compares this header against the cookie of the same value
   (double-submit): a cross-site page can send the request but cannot read
   our cookie, so it cannot set the header. Every write therefore needs it,
   and the server answers 403 without it. Hooked here rather than at each
   call site because this module already owns window.fetch, and there are
   nineteen fetch() calls across the other modules. */
function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)ritrova_csrf=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : null;
}

function requestMethod(resource, init) {
  if (init && init.method) return init.method.toUpperCase();
  if (resource && typeof resource !== 'string' && resource.method) {
    return resource.method.toUpperCase();
  }
  return 'GET';
}

const _origFetch = window.fetch.bind(window);
window.fetch = async function (resource, init) {
  const skip = init && init.skipErrorToast;
  // Strip our custom option before handing to real fetch.
  let realInit = init;
  if (init && 'skipErrorToast' in init) {
    realInit = { ...init };
    delete realInit.skipErrorToast;
  }
  if (MUTATING_METHODS.has(requestMethod(resource, realInit))) {
    const token = csrfToken();
    if (token) {
      realInit = { ...(realInit || {}) };
      const headers = new Headers(realInit.headers || {});
      headers.set(CSRF_HEADER, token);
      realInit.headers = headers;
    }
  }
  try {
    const response = await _origFetch(resource, realInit);
    if (!skip && !response.ok && response.status !== 409) {
      const label = response.status >= 500 ? 'Server error' : 'Request failed';
      safeToast('error', `${label} (${response.status}) — ${_describeRequest(resource)}`);
    }
    return response;
  } catch (err) {
    if (!skip) {
      safeToast('error', `Network error — ${err && err.message ? err.message : 'request aborted'}`);
    }
    throw err;
  }
};

function _describeRequest(resource) {
  try {
    const url = typeof resource === 'string' ? resource : resource.url;
    return url.split('?')[0];
  } catch (_) {
    return 'request';
  }
}

// Exported so sibling modules can toast without touching Alpine directly.
export function safeToast(level, message, opts = {}) {
  const store = window.Alpine && window.Alpine.store && window.Alpine.store('toast');
  if (store) {
    store.show({ ...opts, message, level });
  } else {
    // Pre-init fallback: log so errors aren't lost during page bootstrap.
    // eslint-disable-next-line no-console
    console.warn(`[toast:${level}]`, message);
  }
}

// htmx uses XMLHttpRequest, so the fetch wrapper above doesn't see its
// traffic. Surface both response errors (4xx/5xx) and send errors
// (network / abort) through the same toast store.
document.addEventListener('htmx:responseError', (evt) => {
  const xhr = evt.detail && evt.detail.xhr;
  const status = xhr ? xhr.status : '?';
  if (status === 409) return; // reserved for needs_confirm flows
  safeToast('error', `Request failed (${status})`);
});

document.addEventListener('htmx:sendError', () => {
  safeToast('error', 'Network error — could not reach the server');
});

// htmx uses XMLHttpRequest, so the fetch patch above never sees it. Its
// mutating requests need the same header.
document.addEventListener('htmx:configRequest', (evt) => {
  const verb = (evt.detail.verb || 'get').toUpperCase();
  if (!MUTATING_METHODS.has(verb)) return;
  const token = csrfToken();
  if (token) evt.detail.headers[CSRF_HEADER] = token;
});

export { CSRF_COOKIE, CSRF_HEADER, csrfToken };
