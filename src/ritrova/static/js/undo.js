/* ===================================================================
   undo: FEAT-5 global single-step undo — toast + `z` keybinding
   -------------------------------------------------------------------
   The server registers inverse actions in an in-memory single-slot
   store and hands back an opaque token. We surface that as a toast
   with an Undo button. Two entry points feed it:
     1. HX-Trigger `undoToast` event for htmx-driven forms.
     2. /api/undo/peek on page load — recovers the toast after a
        redirect or a plain fetch that navigates away.
   Clicking Undo POSTs to /api/undo/{token}; on success we reload so
   DOM and DB state realign without per-page bespoke revert logic.

   State shared with alpine-components.js' Alpine.store('toast'):
   when the user dismisses the undo toast (via timer or the `×`
   button), that store must null out the pending token so the `z`
   shortcut doesn't fire a stale undo. We expose `clearUndoStateIfMatches`
   for the toast store to call on dismiss.
   =================================================================== */

const UNDO_TOAST_DURATION_MS = 15000;

// Track the currently-rendered undo toast so we can dismiss it when a newer
// one arrives (server enforces single-slot; mirror that on the client).
let _currentUndoToastId = null;
// Remember the latest token so the `z` keyboard shortcut can fire it without
// scraping the DOM.
let _pendingUndoToken = null;
// Pre-Alpine holding slot for undo toasts that arrive before alpine:init
// (rare, but possible if HX-Trigger fires during a very fast page boot).
let _pendingUndoOnInit = null;

async function _applyUndo(token) {
  if (!token) return;
  try {
    const resp = await fetch(`/api/undo/${token}`, { method: 'POST' });
    if (!resp.ok) return; // global fetch wrapper already surfaced an error toast
    _pendingUndoToken = null;
    // Reload to reconcile the page with restored DB state. Per-page targeted
    // revert could come later — this is the MVP.
    window.location.reload();
  } catch (_) {
    // Network error — global wrapper already toasted.
  }
}

window.showUndoToast = function (opts) {
  if (!opts || !opts.token || !opts.message) return;
  const store = window.Alpine && window.Alpine.store && window.Alpine.store('toast');
  if (!store) {
    // Alpine not up yet — stash and show on init. Rare; see alpine:init below.
    _pendingUndoOnInit = opts;
    return;
  }
  // Dismiss any previously-rendered undo toast: single-slot means the old
  // token was just clobbered server-side and clicking it would 404.
  if (_currentUndoToastId !== null) {
    store.dismiss(_currentUndoToastId);
  }
  _pendingUndoToken = opts.token;
  _currentUndoToastId = store.show({
    message: opts.message,
    level: 'success',
    duration: UNDO_TOAST_DURATION_MS,
    action: {
      label: 'Undo',
      onClick: () => _applyUndo(opts.token),
    },
  });
};

// Called by the Alpine toast store's dismiss() — when the user / timer
// removes the undo toast, we must null the pending token so the `z`
// shortcut doesn't fire a stale undo.
export function clearUndoStateIfMatches(id) {
  if (id === _currentUndoToastId) {
    _pendingUndoToken = null;
    _currentUndoToastId = null;
  }
}

// Called from alpine-components.js on alpine:init to flush any toast that
// arrived before Alpine was ready + trigger the /api/undo/peek recovery.
export function flushPendingUndoOnInit() {
  if (_pendingUndoOnInit) {
    const opts = _pendingUndoOnInit;
    _pendingUndoOnInit = null;
    window.showUndoToast(opts);
    return true;
  }
  return false;
}

// htmx fires a CustomEvent for every key in an HX-Trigger JSON payload.
document.body.addEventListener('undoToast', (evt) => {
  window.showUndoToast(evt.detail);
});

// Keyboard shortcut `z` — design-guide.md reserves it for undo. Ignore when
// typing into a text field so we don't hijack search / rename inputs.
document.addEventListener('keydown', (evt) => {
  if (evt.key !== 'z' || evt.metaKey || evt.ctrlKey || evt.altKey) return;
  const t = evt.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (!_pendingUndoToken) return;
  evt.preventDefault();
  _applyUndo(_pendingUndoToken);
});
