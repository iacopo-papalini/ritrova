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
