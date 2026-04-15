/* ===================================================================
   FaceRecog — Alpine.js components & htmx event handlers
   =================================================================== */

// --------------- Global fetch wrapper: surface server errors ---------------
// Auto-toasts on non-2xx responses and network errors so mutation actions
// never fail silently (BUG-20). Opt-out by setting { skipErrorToast: true }
// on the fetch init object.
//
// 409 Conflict is reserved for "needs_confirm" flows (see BUG-19 — cross-
// species assignment), where the UI deliberately handles the response. We
// skip the auto-toast for 409 so it doesn't fire alongside a confirm dialog.
const _origFetch = window.fetch.bind(window);
window.fetch = async function (resource, init) {
  const skip = init && init.skipErrorToast;
  // Strip our custom option before handing to real fetch (it ignores extras,
  // but we don't want to rely on that).
  let realInit = init;
  if (init && 'skipErrorToast' in init) {
    realInit = { ...init };
    delete realInit.skipErrorToast;
  }
  try {
    const response = await _origFetch(resource, realInit);
    if (!skip && !response.ok && response.status !== 409) {
      const label = response.status >= 500 ? 'Server error' : 'Request failed';
      _safeToast('error', `${label} (${response.status}) — ${_describeRequest(resource)}`);
    }
    return response;
  } catch (err) {
    if (!skip) {
      _safeToast('error', `Network error — ${err && err.message ? err.message : 'request aborted'}`);
    }
    throw err;
  }
};

function _describeRequest(resource) {
  try {
    const url = typeof resource === 'string' ? resource : resource.url;
    // Strip query string for a shorter toast.
    return url.split('?')[0];
  } catch (_) {
    return 'request';
  }
}

// Expose a global helper callable from inline handlers / legacy functions.
// Falls back to console if Alpine hasn't initialized yet.
window.showToast = function (opts) {
  if (typeof opts === 'string') opts = { message: opts };
  _safeToast(opts.level || 'info', opts.message, opts);
};

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

function _safeToast(level, message, opts = {}) {
  const store = window.Alpine && window.Alpine.store && window.Alpine.store('toast');
  if (store) {
    store.show({ ...opts, message, level });
  } else {
    // Pre-init fallback: log so errors aren't lost during page bootstrap.
    // eslint-disable-next-line no-console
    console.warn(`[toast:${level}]`, message);
  }
}

// --------------- FEAT-5: global single-step undo ---------------
// The server registers inverse actions in an in-memory single-slot store and
// hands back an opaque token. We surface that as a toast with an Undo button.
// Two entry points feed it:
//   1. HX-Trigger `undoToast` event for htmx-driven forms.
//   2. /api/undo/peek on page load — recovers the toast after a redirect or
//      a plain fetch that navigates away (the existing cluster call sites
//      all location.href away, so peek covers them).
// Clicking Undo POSTs to /api/undo/{token}; on success we reload so DOM and
// DB state realign without per-page bespoke revert logic (acceptable MVP
// tradeoff — original actions still use surgical DOM updates).
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

// htmx fires a CustomEvent for every key in an HX-Trigger JSON payload.
document.body.addEventListener('undoToast', (evt) => {
  window.showUndoToast(evt.detail);
});

// Keyboard shortcut `z` — design-guide.md reserves it for undo. Ignore when
// typing into a text field so we don't hijack search/rename inputs.
document.addEventListener('keydown', (evt) => {
  if (evt.key !== 'z' || evt.metaKey || evt.ctrlKey || evt.altKey) return;
  const t = evt.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (!_pendingUndoToken) return;
  evt.preventDefault();
  _applyUndo(_pendingUndoToken);
});

// --------------- Alpine: face-selection component ---------------
document.addEventListener('alpine:init', () => {

  // --------------- Alpine store: toast notifications ---------------
  // API:
  //   $store.toast.show({ message, level, action?, duration? }) → id
  //   $store.toast.error(msg) / success(msg) / info(msg)
  //   $store.toast.dismiss(id)
  // `action` shape: { label: string, onClick: () => void }  (future-proofed
  // for FEAT-5 undo: "Assigned 14 faces [Undo]").
  Alpine.store('toast', {
    items: [],
    _nextId: 1,
    show({ message, level = 'info', action = null, duration = 8000 } = {}) {
      const id = this._nextId++;
      this.items.push({ id, message, level, action });
      if (duration > 0) {
        setTimeout(() => this.dismiss(id), duration);
      }
      return id;
    },
    dismiss(id) {
      this.items = this.items.filter((t) => t.id !== id);
    },
    error(message, opts = {}) { return this.show({ ...opts, message, level: 'error' }); },
    success(message, opts = {}) { return this.show({ ...opts, message, level: 'success' }); },
    info(message, opts = {}) { return this.show({ ...opts, message, level: 'info' }); },
  });

  // Surface pre-Alpine undo toast and check for server-side pending undo
  // (e.g. a previous write ended in a redirect; HX-Trigger was lost).
  if (_pendingUndoOnInit) {
    window.showUndoToast(_pendingUndoOnInit);
    _pendingUndoOnInit = null;
  } else {
    fetch('/api/undo/peek', { skipErrorToast: true })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.pending) {
          window.showUndoToast({ message: data.message, token: data.token });
        }
      })
      .catch(() => { /* network blip — no toast is fine */ });
  }

  // --------------- Alpine store: confirmation dialog ---------------
  // Promise-based replacement for browser confirm(). API:
  //   const ok = await $store.dialog.confirm({
  //     title, message?, confirmLabel?, cancelLabel?, danger?
  //   });
  // Escape / backdrop click / cancel button → resolves false.
  // Confirm button → resolves true. If the dialog is already open the
  // caller's promise resolves false immediately (prevents re-entry).
  Alpine.store('dialog', {
    open: false,
    title: '',
    message: '',
    confirmLabel: 'Confirm',
    cancelLabel: 'Cancel',
    danger: false,
    _resolve: null,

    confirm(opts = {}) {
      if (this.open) return Promise.resolve(false);
      this.title = opts.title || 'Are you sure?';
      this.message = opts.message || '';
      this.confirmLabel = opts.confirmLabel || 'Confirm';
      this.cancelLabel = opts.cancelLabel || 'Cancel';
      this.danger = opts.danger === true;
      this.open = true;
      return new Promise((resolve) => { this._resolve = resolve; });
    },

    _settle(result) {
      const r = this._resolve;
      this._resolve = null;
      this.open = false;
      if (r) r(result);
    },

    ok() { this._settle(true); },
    cancel() { this._settle(false); },
  });


  Alpine.data('faceSelection', () => ({
    selected: new Set(),

    get count() { return this.selected.size; },
    get hasSelection() { return this.selected.size > 0; },

    toggle(id) {
      this.selected.has(id)
        ? this.selected.delete(id)
        : this.selected.add(id);
    },

    selectAll(ids) {
      ids.forEach(id => this.selected.add(id));
    },

    clear() {
      this.selected.clear();
    },

    get ids() { return [...this.selected]; }
  }));

  // --------------- Alpine: subject picker (typeahead) ---------------
  // Usage: x-data="subjectPicker({ multi: false, onSelect(subject) { ... } })"
  //   or:  x-data="subjectPicker({ multi: true })"  then read .selected
  let _subjectsCache = null;

  Alpine.data('subjectPicker', (opts = {}) => ({
    query: '',
    items: [],
    open: false,
    loading: true,
    selected: [],        // multi-select: array of {id, name}
    multi: opts.multi || false,
    allowCreate: opts.allowCreate !== false, // default true, set false to disable
    hi: -1,              // highlighted index in dropdown

    async init() {
      if (!_subjectsCache) {
        const r = await fetch('/api/subjects/all');
        _subjectsCache = await r.json();
      }
      this.items = _subjectsCache;
      this.loading = false;
    },

    get filtered() {
      const q = this.query.toLowerCase().trim();
      if (!q) return this.items.slice(0, 10);
      return this.items.filter(p => p.name.toLowerCase().includes(q)).slice(0, 10);
    },

    get showCreate() {
      if (!this.allowCreate) return false;
      const q = this.query.trim();
      return q && !this.items.some(p => p.name.toLowerCase() === q.toLowerCase());
    },

    // Keyboard: total selectable slots (filtered items + optional create)
    get _totalSlots() {
      return this.filtered.length + (this.showCreate ? 1 : 0);
    },

    onArrowDown() {
      if (!this.open) { this.open = true; this.hi = 0; return; }
      this.hi = (this.hi + 1) % this._totalSlots;
      this._scrollToHighlighted();
    },

    onArrowUp() {
      if (!this.open) return;
      this.hi = (this.hi - 1 + this._totalSlots) % this._totalSlots;
      this._scrollToHighlighted();
    },

    onEnter() {
      if (!this.open || this.hi < 0) return;
      if (this.hi < this.filtered.length) {
        this.pick(this.filtered[this.hi]);
      } else if (this.showCreate) {
        this.create();
      }
    },

    _scrollToHighlighted() {
      this.$nextTick(() => {
        const el = this.$refs.dropdown?.querySelector('[data-hi="true"]');
        if (el) el.scrollIntoView({ block: 'nearest' });
      });
    },

    pick(subject) {
      this.hi = -1;
      if (this.multi) {
        const idx = this.selected.findIndex(s => s.id === subject.id);
        if (idx >= 0) this.selected.splice(idx, 1);
        else this.selected.push(subject);
        this.query = '';
      } else {
        this.query = subject.name;
        this.open = false;
        if (opts.onSelect) opts.onSelect.call(this, subject);
        this.$dispatch('subject-selected', subject);
      }
    },

    isSelected(id) {
      return this.selected.some(s => s.id === id);
    },

    removeSelected(id) {
      this.selected = this.selected.filter(s => s.id !== id);
    },

    async create() {
      const name = this.query.trim();
      if (!name) return;
      const r = await fetch('/api/subjects/create', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
      });
      if (!r.ok) return;
      const subject = await r.json();
      // Invalidate cache
      _subjectsCache = null;
      await this.init();
      this.pick(subject);
    }
  }));

  // --------------- Alpine: lightbox store ---------------
  // The lightbox unit is an *item* — either a finding (video frame or photo
  // crop source) or a source. Callers can open a single item or a list; with
  // a list, arrow keys navigate (FEAT-18). Finding items resolve images via
  // /api/findings/{id}/frame so video-frame findings render correctly
  // (FEAT-14) — previously the store 404'd on video sources.
  //
  // Back-compat: `show(sourceId)` still works. New callers prefer
  // `openFromGrid($el, 'finding')` which walks the DOM for sibling thumbnails.
  Alpine.store('lightbox', {
    open: false,
    items: [],      // [{ type: 'finding'|'source', id: number }]
    index: 0,
    // Populated from the /info endpoint for the current item.
    sourceId: null,
    photoPath: null,
    latitude: null,
    longitude: null,
    type: null,     // 'photo' | 'video' | null (while loading)
    rotation: 0,
    // Back-compat accessor for templates written against the old store.
    get photoId() {
      const it = this.items[this.index];
      if (!it) return null;
      return it.type === 'source' ? it.id : this.sourceId;
    },

    // ----- Computed URLs (reactive via the items/index state) -----
    get currentItem() { return this.items[this.index] || null; },
    get imageUrl() {
      const it = this.currentItem;
      if (!it) return '';
      return it.type === 'finding'
        ? `/api/findings/${it.id}/frame?max_size=1600`
        : `/api/sources/${it.id}/image?max_size=1600`;
    },
    get detailUrl() { return this.sourceId ? `/photos/${this.sourceId}` : ''; },
    get downloadUrl() { return this.sourceId ? `/api/sources/${this.sourceId}/original` : ''; },
    get hasNav() { return this.items.length > 1; },
    get canPrev() { return this.index > 0; },
    get canNext() { return this.index < this.items.length - 1; },

    // ----- Public API -----
    // Legacy single-source entry. Kept so external callers don't break.
    show(sourceId) { this._openList([{ type: 'source', id: sourceId }], 0); },
    showSource(sourceId) { this._openList([{ type: 'source', id: sourceId }], 0); },
    showFinding(findingId) { this._openList([{ type: 'finding', id: findingId }], 0); },
    openSources(ids, start = 0) {
      this._openList(ids.map((id) => ({ type: 'source', id })), start);
    },
    openFindings(ids, start = 0) {
      this._openList(ids.map((id) => ({ type: 'finding', id })), start);
    },

    // Derive the sibling list from the DOM at click time — robust to htmx
    // pagination because the DOM is authoritative when the user clicks.
    // Expects thumbnails to carry data-finding-id (or data-source-id) and an
    // ancestor with data-lightbox-group (falls back to immediate parent).
    openFromGrid(el, mode = 'finding') {
      const attr = mode === 'finding' ? 'data-finding-id' : 'data-source-id';
      const thumb = el.closest(`[${attr}]`);
      if (!thumb) return;
      const group = thumb.closest('[data-lightbox-group]') || thumb.parentElement;
      if (!group) return;
      const nodes = Array.from(group.querySelectorAll(`[${attr}]`));
      const ids = nodes.map((n) => Number(n.getAttribute(attr)));
      const start = Math.max(0, nodes.indexOf(thumb));
      if (mode === 'finding') this.openFindings(ids, start);
      else this.openSources(ids, start);
    },

    next() { if (this.canNext) this._goto(this.index + 1); },
    prev() { if (this.canPrev) this._goto(this.index - 1); },
    rotate() { this.rotation = (this.rotation + 90) % 360; },

    close() {
      this.open = false;
      this.items = [];
      this.index = 0;
      this._resetInfo();
    },

    // ----- Internals -----
    _openList(items, start) {
      if (!items.length) return;
      this.items = items;
      this.index = Math.min(Math.max(0, start), items.length - 1);
      this._resetInfo();
      this.open = true;
      this._fetchInfo();
    },

    _goto(newIndex) {
      this.index = newIndex;
      this._resetInfo();
      this._fetchInfo();
    },

    _resetInfo() {
      this.sourceId = null;
      this.photoPath = null;
      this.latitude = null;
      this.longitude = null;
      this.type = null;
      this.rotation = 0;
    },

    _fetchInfo() {
      const it = this.currentItem;
      if (!it) return;
      const url = it.type === 'finding'
        ? `/api/findings/${it.id}/info`
        : `/api/sources/${it.id}/info`;
      const targetIndex = this.index;
      fetch(url)
        .then((r) => r.ok ? r.json() : null)
        .then((data) => {
          // Ignore late responses if the user navigated on while the request was in flight.
          if (!data || targetIndex !== this.index) return;
          this.sourceId = data.source_id != null ? data.source_id : it.id;
          this.photoPath = data.file_path;
          this.latitude = data.latitude;
          this.longitude = data.longitude;
          this.type = data.type || null;
        });
    },
  });

  // --------------- Alpine: video player store ---------------
  // Full-screen overlay with a native <video controls autoplay>. Lighter than
  // the lightbox (no nav, no rotate) — distinct concern, distinct store.
  Alpine.store('videoPlayer', {
    open: false,
    sourceId: null,
    sourceName: '',
    show(sourceId, sourceName = '') {
      this.sourceId = sourceId;
      this.sourceName = sourceName;
      this.open = true;
    },
    close() {
      this.open = false;
      this.sourceId = null;
      this.sourceName = '';
    },
  });
});

// --------------- htmx: global error handlers ---------------
// htmx uses XMLHttpRequest, so the global fetch wrapper above doesn't see
// its traffic. We surface both response errors (4xx/5xx) and send errors
// (network/abort) via the same toast store.
document.addEventListener('htmx:responseError', (evt) => {
  const xhr = evt.detail && evt.detail.xhr;
  const status = xhr ? xhr.status : '?';
  if (status === 409) return; // reserved for needs_confirm flows
  _safeToast('error', `Request failed (${status})`);
});

document.addEventListener('htmx:sendError', () => {
  _safeToast('error', 'Network error — could not reach the server');
});

// --------------- Legacy helpers (kept for backward compat) ---------------
// Errors no longer silently swallowed — the global fetch wrapper above
// surfaces any non-2xx (except 409) and network failures as toasts.
async function assignFinding(event, faceId) {
  event.preventDefault();
  const form = event.target;
  const select = form.querySelector('select');
  const personId = select.value;
  if (!personId) return false;

  const body = new FormData();
  body.append('person_id', personId);

  const resp = await fetch(`/api/findings/${faceId}/assign`, {method: 'POST', body});
  if (resp.ok) {
    location.reload();
  }
  return false;
}

async function unassignFinding(faceId) {
  const resp = await fetch(`/api/findings/${faceId}/unassign`, {method: 'POST'});
  if (resp.ok) {
    location.reload();
  }
}
