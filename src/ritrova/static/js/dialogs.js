/* ===================================================================
   dialogs: promise-based confirm dialog + claim-faces helper
   -------------------------------------------------------------------
   window.confirmDialog — replaces browser confirm(), talks to the
   Alpine 'dialog' store (defined in alpine-components.js).

   window.claimFaces — handles the 409 "different species" needs_confirm
   flow by prompting the user via confirmDialog and retrying with
   force=true.
   =================================================================== */

// Promise-based confirmation dialog — replaces browser confirm().
// Usage:  if (!(await confirmDialog({ title, message, danger }))) return;
window.confirmDialog = function (opts) {
  const store = window.Alpine && window.Alpine.store && window.Alpine.store('dialog');
  if (!store) {
    // Pre-init fallback: native confirm so actions aren't lost during bootstrap.
    // eslint-disable-next-line no-alert
    return Promise.resolve(window.confirm((opts && (opts.title || opts.message)) || 'Are you sure?'));
  }
  return store.confirm(opts || {});
};

// Claim one-or-more findings for a subject. Handles the 409 "different species"
// needs_confirm flow with a confirm dialog and a force=true retry. Returns
// Promise<boolean> — true if claimed, false if the user cancelled or the request
// failed (the global fetch wrapper already toasted non-2xx / non-409).
window.claimFaces = async function (subjectId, faceIds, opts = {}) {
  const body = { face_ids: faceIds, force: !!opts.force };
  const resp = await fetch(`/api/subjects/${subjectId}/claim-faces`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (resp.ok) return true;
  if (resp.status !== 409) return false;
  const data = await resp.json().catch(() => ({}));
  if (!data.needs_confirm) return false;
  const ok = await window.confirmDialog({
    title: 'Different species',
    message: (data.error || 'Species mismatch') + '\n\nAssign anyway and correct the species?',
    confirmLabel: 'Assign',
  });
  if (!ok) return false;
  return window.claimFaces(subjectId, faceIds, { force: true });
};
