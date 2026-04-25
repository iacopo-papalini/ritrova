/* ===================================================================
   alpine-components: all Alpine stores + x-data factories
   -------------------------------------------------------------------
   Registered on 'alpine:init' so templates can reference them via
   x-data / $store.
   =================================================================== */

import { clearUndoStateIfMatches, flushPendingUndoOnInit } from './undo.js';

document.addEventListener('alpine:init', () => {

  // --------------- Alpine store: toast notifications ---------------
  // API:
  //   $store.toast.show({ message, level, action?, duration? }) → id
  //   $store.toast.error(msg) / success(msg) / info(msg)
  //   $store.toast.dismiss(id)
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
      // If the dismissed toast was the live undo toast, clear pending token
      // so the `z` shortcut doesn't fire a stale undo.
      clearUndoStateIfMatches(id);
    },
    error(message, opts = {}) { return this.show({ ...opts, message, level: 'error' }); },
    success(message, opts = {}) { return this.show({ ...opts, message, level: 'success' }); },
    info(message, opts = {}) { return this.show({ ...opts, message, level: 'info' }); },
  });

  // Surface pre-Alpine undo toast and check for server-side pending undo
  // (e.g. a previous write ended in a redirect; HX-Trigger was lost).
  if (!flushPendingUndoOnInit()) {
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


  // --------------- Alpine: face-selection component ---------------
  Alpine.data('faceSelection', () => ({
    selected: new Set(),
    anchor: null,         // last plain-clicked id, pivot for shift-click range

    // Drag state — underscored because it's not meant to be read from templates.
    _dragging: false,
    _dragAction: 'add',   // 'add' or 'remove' — decided by the start tile's initial state
    _dragStartId: null,
    _dragMoved: false,    // set once the pointer enters a different tile
    _dragLastId: null,

    init() {
      const root = this.$root;
      root.addEventListener('mousedown', (e) => this._onMouseDown(e));
      root.addEventListener('mouseover', (e) => this._onMouseOver(e));
      // Face thumbnails are <img>s — by default the browser starts a native
      // drag when you press on them, which eats mouseover on sibling tiles
      // and kills our drag-select. Veto dragstart inside the grid.
      root.addEventListener('dragstart', (e) => {
        if (e.target.closest('[data-finding-id]')) e.preventDefault();
      });
      // Listen on window so releases outside the grid still end the drag.
      this._onUp = () => { this._dragging = false; this._dragLastId = null; };
      window.addEventListener('mouseup', this._onUp);
    },

    _tileIdFromEvent(e) {
      const el = e.target.closest('[data-finding-id]');
      if (!el || !this.$root.contains(el)) return null;
      const id = parseInt(el.dataset.findingId, 10);
      return Number.isNaN(id) ? null : id;
    },

    _onMouseDown(e) {
      // Left button only; don't start drag when grabbing a button/link inside a tile.
      if (e.button !== 0 || e.target.closest('button, a')) return;
      const id = this._tileIdFromEvent(e);
      if (id === null) return;
      this._dragging = true;
      this._dragStartId = id;
      this._dragMoved = false;
      this._dragLastId = id;
      this._dragAction = this.selected.has(id) ? 'remove' : 'add';
      // Suppress text selection & native image-drag so mouseover keeps firing.
      e.preventDefault();
    },

    _onMouseOver(e) {
      if (!this._dragging) return;
      const id = this._tileIdFromEvent(e);
      if (id === null || id === this._dragLastId) return;
      this._dragLastId = id;
      if (!this._dragMoved) {
        // First real drag move — apply to the start tile retroactively.
        this._dragMoved = true;
        this._apply(this._dragStartId);
      }
      this._apply(id);
    },

    _apply(id) {
      if (this._dragAction === 'add') this.selected.add(id);
      else this.selected.delete(id);
    },

    // Called from tile `@click` — same element mousedown+mouseup only.
    tileClick(id, event) {
      // A drag that moved already selected/deselected the end tile; don't toggle again.
      if (this._dragMoved) { this._dragMoved = false; return; }
      if (event && event.shiftKey && this.anchor !== null && this.anchor !== id) {
        this._extendTo(id);
        return;
      }
      this.toggle(id);
      this.anchor = id;
    },

    _extendTo(id) {
      // Range-select from anchor to id in DOM (reading) order; additive only.
      const ids = [...this.$root.querySelectorAll('[data-finding-id]')]
        .map(el => parseInt(el.dataset.findingId, 10));
      const i0 = ids.indexOf(this.anchor);
      const i1 = ids.indexOf(id);
      if (i0 < 0 || i1 < 0) return;
      const [a, b] = i0 < i1 ? [i0, i1] : [i1, i0];
      for (let i = a; i <= b; i++) this.selected.add(ids[i]);
      this.anchor = id;
    },

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
      this.anchor = null;
    },

    get ids() { return [...this.selected]; }
  }));

  // --------------- Alpine: manual-finding (FEAT-29) ---------------
  // Shift+drag on the photo viewer to draw a bbox, species popover,
  // POST /api/sources/{sourceId}/findings, reload on success so the
  // newly-rendered photo page picks up the finding plus its prefilled
  // subject picker.
  //
  // Usage: x-data="manualFinding({ sourceId, defaultSpecies })" on the
  // <div class="relative inline-block ..."> photo wrapper.
  Alpine.data('manualFinding', (opts = {}) => ({
    sourceId: opts.sourceId,
    // Real source pixel dimensions (from the DB-stored source row). The
    // served /api/sources/{id}/image is downscaled to max_size=1600, so
    // img.naturalWidth is NOT the source size — we must scale the drawn
    // bbox against the real source dims, not the served image dims.
    sourceWidth: opts.sourceWidth || 0,
    sourceHeight: opts.sourceHeight || 0,
    // Default species for the popover; can be overridden by the user.
    // 'human' on /people/, 'dog' on /pets/ (cat toggle available).
    defaultSpecies: opts.defaultSpecies || 'human',
    species: opts.defaultSpecies || 'human',

    // Drag state — all in rendered-image pixels relative to the img element.
    dragging: false,
    startX: 0,
    startY: 0,
    curX: 0,
    curY: 0,

    // Popover state.
    popoverOpen: false,
    popoverX: 0,
    popoverY: 0,

    // Busy flag (suppresses double-submit while fetch is inflight).
    submitting: false,

    init() {
      this._img = this.$root.querySelector('img#photo-img');
      if (!this._img) return;
      this._onDown = (e) => this._onMouseDown(e);
      this._onMove = (e) => this._onMouseMove(e);
      this._onUp = (e) => this._onMouseUp(e);
      this._img.addEventListener('mousedown', this._onDown);
      window.addEventListener('mousemove', this._onMove);
      window.addEventListener('mouseup', this._onUp);
    },

    destroy() {
      if (this._img) this._img.removeEventListener('mousedown', this._onDown);
      window.removeEventListener('mousemove', this._onMove);
      window.removeEventListener('mouseup', this._onUp);
    },

    _imgRect() {
      return this._img.getBoundingClientRect();
    },

    _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); },

    _localXY(clientX, clientY) {
      const r = this._imgRect();
      return {
        x: this._clamp(clientX - r.left, 0, r.width),
        y: this._clamp(clientY - r.top, 0, r.height),
      };
    },

    _onMouseDown(e) {
      // Block new drags while a previous submit is still in flight — the
      // server-side ArcFace/SigLIP pass takes 1-3s; dragging again during
      // that window would just queue a second popover with no benefit.
      if (this.submitting) return;
      if (!e.shiftKey || e.button !== 0) return;
      // Prevent native image-drag and text selection.
      e.preventDefault();
      const { x, y } = this._localXY(e.clientX, e.clientY);
      this.dragging = true;
      this.startX = x;
      this.startY = y;
      this.curX = x;
      this.curY = y;
      this.popoverOpen = false;
    },

    _onMouseMove(e) {
      if (!this.dragging) return;
      const { x, y } = this._localXY(e.clientX, e.clientY);
      this.curX = x;
      this.curY = y;
    },

    _onMouseUp(e) {
      if (!this.dragging) return;
      this.dragging = false;
      // If release happened outside the image bounds, the local coords
      // are already clamped — but we also veto drags smaller than a few
      // pixels (probably an accidental shift-click).
      const rect = this.rect;
      if (rect.w < 5 || rect.h < 5) {
        return;
      }
      // Position the popover just below the bottom-right of the rect,
      // falling back upward if it would clip the viewport.
      const imgRect = this._imgRect();
      const px = imgRect.left + rect.x + rect.w + 8;
      const py = imgRect.top + rect.y + rect.h + 8;
      this.popoverX = px + window.scrollX;
      this.popoverY = py + window.scrollY;
      // Reset species to the default each time — avoids stale state.
      this.species = this.defaultSpecies;
      this.popoverOpen = true;
    },

    // Rendered-pixel rectangle (for the live SVG / DOM overlay).
    get rect() {
      const x = Math.min(this.startX, this.curX);
      const y = Math.min(this.startY, this.curY);
      const w = Math.abs(this.curX - this.startX);
      const h = Math.abs(this.curY - this.startY);
      return { x, y, w, h };
    },

    // Translate rendered-pixel rect to *source* pixels. Use the DB-stored
    // source dims (not img.naturalWidth/Height) because the served image
    // may have been downscaled by /api/sources/{id}/image (max_size=1600).
    // Fall back to naturalWidth/Height only if the server didn't provide
    // source dims (very old source rows with width=0).
    _sourceBbox() {
      const img = this._img;
      const srcW = this.sourceWidth > 0 ? this.sourceWidth : img.naturalWidth;
      const srcH = this.sourceHeight > 0 ? this.sourceHeight : img.naturalHeight;
      const scaleX = srcW / img.clientWidth;
      const scaleY = srcH / img.clientHeight;
      const { x, y, w, h } = this.rect;
      return [
        Math.round(x * scaleX),
        Math.round(y * scaleY),
        Math.round(w * scaleX),
        Math.round(h * scaleY),
      ];
    },

    cancel() {
      this.popoverOpen = false;
    },

    async confirm() {
      if (this.submitting) return;
      const bbox = this._sourceBbox();
      if (bbox[2] < 20 || bbox[3] < 20) {
        if (window.Alpine && Alpine.store('toast')) {
          Alpine.store('toast').error('Drag a larger rectangle — minimum 20×20 source pixels.');
        }
        this.popoverOpen = false;
        return;
      }
      this.submitting = true;
      // Close the popover immediately — ArcFace/SigLIP takes 1-3s and the
      // user needs feedback that their click registered. A persistent info
      // toast takes the popover's place as the "something's happening"
      // signal and is dismissed when the response comes back (or the page
      // navigates away on success).
      this.popoverOpen = false;
      const toastStore = window.Alpine && Alpine.store('toast');
      const busyToastId = toastStore
        ? toastStore.show({ message: 'Adding face…', level: 'info', duration: 0 })
        : null;
      try {
        const resp = await fetch(`/api/sources/${this.sourceId}/findings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bbox, species: this.species }),
        });
        if (!resp.ok) return; // fetch-wrapper surfaces the error toast.
        const data = await resp.json();
        if (data.undo_token) {
          window.showUndoToast({ message: data.message, token: data.undo_token });
        }
        // Reload so the server-rendered face grid + overlays include the
        // new finding. Hash focuses the new tile; query hint prefills
        // the subject picker via the server-rendered photo page. See
        // FEAT-29 spec — acceptable to reload for MVP.
        const suggest = data.suggestion ? `&suggest=${encodeURIComponent(data.suggestion.name)}` : '';
        const url = `${window.location.pathname}?focus=${data.finding_id}${suggest}#face-${data.finding_id}`;
        window.location.href = url;
      } finally {
        this.submitting = false;
        if (busyToastId !== null && toastStore) toastStore.dismiss(busyToastId);
      }
    },
  }));

  // --------------- Alpine: subject picker (typeahead) ---------------
  // Usage: x-data="subjectPicker({ multi: false, onSelect(subject) { ... } })"
  //   or:  x-data="subjectPicker({ multi: true })"  then read .selected
  let _subjectsCache = null;

  Alpine.data('subjectPicker', (opts = {}) => ({
    query: opts.initialQuery || '',
    items: [],
    open: false,
    loading: true,
    selected: [],        // multi-select: array of {id, name}
    multi: opts.multi || false,
    // DB-side subject kind ('person'/'pet') used when creating a new subject
    // from the picker. Defaults to 'person' to match the server default.
    // Caller must pass 'pet' on /pets/* pages to avoid a species mismatch
    // 409 when the new subject is then assigned.
    createKind: opts.createKind || 'person',
    allowCreate: opts.allowCreate !== false, // default true, set false to disable
    hi: -1,              // highlighted index in dropdown

    async init() {
      if (!_subjectsCache) {
        const r = await fetch('/api/subjects/all');
        _subjectsCache = await r.json();
      }
      this.items = _subjectsCache;
      this.loading = false;
      // FEAT-29 prefill: if caller passed initialQuery, open the dropdown
      // so the suggested match is visible and Enter confirms immediately.
      if (this.query) {
        this.open = true;
        this.hi = 0;
      }
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
        body: JSON.stringify({name, kind: this.createKind})
      });
      if (!r.ok) return;
      const subject = await r.json();
      // Invalidate cache
      _subjectsCache = null;
      await this.init();
      this.pick(subject);
    }
  }));

  // --------------- Alpine: circle picker (typeahead) ---------------
  // Shape-matches subjectPicker so partials/circle_picker.html can mirror
  // partials/subject_picker.html without a parallel keyboard-nav rewrite.
  //
  // Usage: x-data="circlePicker({ excludeIds: [...], onSelect(circle) { ... } })"
  let _circlesCache = null;

  Alpine.data('circlePicker', (opts = {}) => ({
    query: '',
    items: [],
    open: false,
    loading: true,
    hi: -1,
    excludeIds: new Set(opts.excludeIds || []),

    async init() {
      if (!_circlesCache) {
        const r = await fetch('/api/circles/all');
        const data = await r.json();
        _circlesCache = data.circles;
      }
      this.items = _circlesCache.filter(c => !this.excludeIds.has(c.id));
      this.loading = false;
    },

    get filtered() {
      const q = this.query.toLowerCase().trim();
      if (!q) return this.items.slice(0, 10);
      return this.items.filter(c => c.name.toLowerCase().includes(q)).slice(0, 10);
    },

    get _totalSlots() { return this.filtered.length; },

    onArrowDown() {
      if (!this.open) { this.open = true; this.hi = 0; return; }
      if (this._totalSlots === 0) return;
      this.hi = (this.hi + 1) % this._totalSlots;
      this._scrollToHighlighted();
    },

    onArrowUp() {
      if (!this.open || this._totalSlots === 0) return;
      this.hi = (this.hi - 1 + this._totalSlots) % this._totalSlots;
      this._scrollToHighlighted();
    },

    onEnter() {
      if (!this.open || this.hi < 0 || this.hi >= this.filtered.length) return;
      this.pick(this.filtered[this.hi]);
    },

    _scrollToHighlighted() {
      this.$nextTick(() => {
        const el = this.$refs.dropdown?.querySelector('[data-hi="true"]');
        if (el) el.scrollIntoView({ block: 'nearest' });
      });
    },

    pick(circle) {
      this.hi = -1;
      this.query = circle.name;
      this.open = false;
      if (opts.onSelect) opts.onSelect.call(this, circle);
    },

    invalidateCache() { _circlesCache = null; }
  }));

  // --------------- Alpine: lightbox store ---------------
  // The lightbox unit is an *item* — either a finding (video frame or photo
  // crop source) or a source. Callers can open a single item or a list; with
  // a list, arrow keys navigate (FEAT-18). Finding items resolve images via
  // /api/findings/{id}/frame so video-frame findings render correctly
  // (FEAT-14) — previously the store 404'd on video sources.
  //
  // Entry points: `openFromGrid($el, type)` (preferred — walks the DOM for
  // sibling thumbnails), `showFinding(id)` / `showSource(id)` for singletons.
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

    // ----- Computed URLs (reactive via the items/index state) -----
    get currentItem() { return this.items[this.index] || null; },
    get imageUrl() {
      const it = this.currentItem;
      if (!it) return '';
      return it.type === 'finding'
        ? `/api/findings/${it.id}/frame?max_size=1600`
        : `/api/sources/${it.id}/image?max_size=1600`;
    },
    // Video sources have no meaningful photo-detail page — the /photos/{id}
    // route renders one extracted frame with bbox overlays from every finding
    // in the clip, which looks wrong. Hide the button until a dedicated
    // video-source page exists. See BUG-22.
    get detailUrl() {
      return this.sourceId && this.type !== 'video' ? `/photos/${this.sourceId}` : '';
    },
    get downloadUrl() { return this.sourceId ? `/api/sources/${this.sourceId}/original` : ''; },
    get hasNav() { return this.items.length > 1; },
    get canPrev() { return this.index > 0; },
    get canNext() { return this.index < this.items.length - 1; },

    // ----- Public API -----
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
