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

  // --------------- Alpine: person picker (typeahead) ---------------
  // Usage: x-data="personPicker({ multi: false, onSelect(person) { ... } })"
  //   or:  x-data="personPicker({ multi: true })"  then read .selected
  let _personsCache = null;

  Alpine.data('personPicker', (opts = {}) => ({
    query: '',
    items: [],
    open: false,
    selected: [],        // multi-select: array of {id, name}
    multi: opts.multi || false,

    async init() {
      if (!_personsCache) {
        const r = await fetch('/api/persons/all');
        _personsCache = await r.json();
      }
      this.items = _personsCache;
    },

    get filtered() {
      const q = this.query.toLowerCase().trim();
      if (!q) return this.items.slice(0, 10);
      return this.items.filter(p => p.name.toLowerCase().includes(q)).slice(0, 10);
    },

    get showCreate() {
      const q = this.query.trim();
      return q && !this.items.some(p => p.name.toLowerCase() === q.toLowerCase());
    },

    pick(person) {
      if (this.multi) {
        const idx = this.selected.findIndex(s => s.id === person.id);
        if (idx >= 0) this.selected.splice(idx, 1);
        else this.selected.push(person);
        this.query = '';
      } else {
        this.query = person.name;
        this.open = false;
        if (opts.onSelect) opts.onSelect.call(this, person);
        this.$dispatch('person-selected', person);
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
      const r = await fetch('/api/persons/create', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
      });
      if (!r.ok) return;
      const person = await r.json();
      // Invalidate cache
      _personsCache = null;
      await this.init();
      this.pick(person);
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
      fetch(`/api/photos/${photoId}/info`)
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
async function assignFace(event, faceId) {
  event.preventDefault();
  const form = event.target;
  const select = form.querySelector('select');
  const personId = select.value;
  if (!personId) return false;

  const body = new FormData();
  body.append('person_id', personId);

  const resp = await fetch(`/api/faces/${faceId}/assign`, {method: 'POST', body});
  if (resp.ok) {
    location.reload();
  }
  return false;
}

async function unassignFace(faceId) {
  const resp = await fetch(`/api/faces/${faceId}/unassign`, {method: 'POST'});
  if (resp.ok) {
    location.reload();
  }
}
