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

   window.showToast and window.safeToast are exposed as globals so
   legacy inline handlers and non-module scripts can still use them.
   =================================================================== */

const _origFetch = window.fetch.bind(window);
window.fetch = async function (resource, init) {
  const skip = init && init.skipErrorToast;
  // Strip our custom option before handing to real fetch.
  let realInit = init;
  if (init && 'skipErrorToast' in init) {
    realInit = { ...init };
    delete realInit.skipErrorToast;
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
// Window-global form kept for legacy inline handlers.
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

// Legacy global — inline handlers still call window.showToast(...).
window.showToast = function (opts) {
  if (typeof opts === 'string') opts = { message: opts };
  safeToast(opts.level || 'info', opts.message, opts);
};

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
