/* ===================================================================
   FaceRecog — Alpine.js components & htmx event handlers
   =================================================================== */

// --------------- Alpine: face-selection component ---------------
document.addEventListener('alpine:init', () => {

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
  Alpine.store('lightbox', {
    open: false,
    photoId: null,
    photoPath: null,
    latitude: null,
    longitude: null,
    rotation: 0,

    show(photoId) {
      this.photoId = photoId;
      this.photoPath = null;
      this.latitude = null;
      this.longitude = null;
      this.rotation = 0;
      this.open = true;
      fetch(`/api/sources/${photoId}/info`)
        .then(r => r.json())
        .then(data => {
          this.photoPath = data.file_path;
          this.latitude = data.latitude;
          this.longitude = data.longitude;
        });
    },

    rotate() { this.rotation = (this.rotation + 90) % 360; },

    close() {
      this.open = false;
      this.photoId = null;
      this.photoPath = null;
      this.latitude = null;
      this.longitude = null;
      this.rotation = 0;
    }
  });
});

// --------------- htmx: global error handler ---------------
document.addEventListener('htmx:responseError', (evt) => {
  console.error('htmx error:', evt.detail);
});

// --------------- Legacy helpers (kept for backward compat) ---------------
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
