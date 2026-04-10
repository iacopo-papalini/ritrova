# Frontend Redesign: Tailwind CSS + htmx + Alpine.js

## Goal

Replace 474 lines of custom CSS and 386 lines of inline JS with a modern stack:
- **Tailwind CSS** (CLI-compiled) for styling — dark gallery aesthetic
- **htmx 2.0** for server communication (replaces all fetch/reload patterns)
- **Alpine.js 3.x** for client-side state (face selection, toggles)

Add missing features: infinite scroll on singletons/person_detail, lightbox for thumbnails, loading indicators, error handling.

## Design Direction

**Dark gallery** — personal photo archive feel, not admin panel.

```
Background:  #0f1114 (near-black)
Cards:       #1a1d23 (dark slate)
Accent:      #d4a574 (warm amber)
Text:        #e8e4df (warm white)
Muted:       #6b7280 (gray)
Borders:     #2a2d35 (subtle)
Success:     #4ade80 (green)
Danger:      #ef4444 (red)
```

- Nav: minimal dark bar, logo in amber
- Face grids: borderless thumbnails, tight gaps, photos float on dark background
- Lightbox: full-black backdrop, click any thumbnail to see full image
- Selection: amber ring (not red) on selected faces
- Cluster cards: dark cards, face mosaic fills the card, info overlays bottom

## Tech Setup

### Tailwind CLI (no npm project needed)

```bash
# One-time: download standalone CLI
curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
chmod +x tailwindcss-macos-arm64
mv tailwindcss-macos-arm64 ./tailwindcss

# Build CSS:
./tailwindcss -i src/face_recog/static/input.css -o src/face_recog/static/style.css --minify
```

### input.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body { @apply bg-gallery-bg text-gallery-text; }
}

@layer components {
  /* Lightbox, face-hint hover, etc. — custom components */
}
```

### tailwind.config.js

```js
module.exports = {
  content: ["./src/face_recog/templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        gallery: {
          bg: '#0f1114',
          card: '#1a1d23',
          accent: '#d4a574',
          text: '#e8e4df',
          muted: '#6b7280',
          border: '#2a2d35',
        }
      },
      fontFamily: {
        display: ['"DM Sans"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
      }
    }
  }
}
```

### Makefile

```makefile
.DEFAULT_GOAL := help

.PHONY: help serve scan scan-pets scan-videos cluster auto-assign cleanup css css-watch

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

serve: ## Start the web UI
	uv run face-recog serve

scan: ## Scan photos for human faces
	uv run face-recog scan

scan-pets: ## Scan photos for dogs and cats
	uv run face-recog scan-pets

scan-videos: ## Scan videos for human faces
	uv run face-recog scan-videos

cluster: ## Cluster all faces (humans + pets)
	uv run face-recog cluster

auto-assign: ## Bulk-assign clusters to known persons
	uv run face-recog auto-assign

cleanup: ## Dismiss tiny and blurry faces
	uv run face-recog cleanup

stats: ## Show database statistics
	uv run face-recog stats

css: ## Build Tailwind CSS (one-time)
	./tailwindcss -i src/face_recog/static/input.css -o src/face_recog/static/style.css --minify

css-watch: ## Watch and rebuild CSS on changes
	./tailwindcss -i src/face_recog/static/input.css -o src/face_recog/static/style.css --watch

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linter + type checker
	uv run ruff check src/ tests/ && uv run mypy src/ tests/

migrate-paths: ## Rewrite DB paths to relative
	uv run face-recog migrate-paths
```

---

## File Structure

### New files
- `tailwind.config.js` — Tailwind theme (dark gallery colors, fonts)
- `src/face_recog/static/input.css` — Tailwind entry point
- `Makefile` — developer commands
- `src/face_recog/templates/partials/face_thumb.html` — reusable face thumbnail
- `src/face_recog/templates/partials/face_grid.html` — page of thumbnails (infinite scroll)
- `src/face_recog/templates/partials/cluster_card.html` — cluster card with hint
- `src/face_recog/templates/partials/cluster_hint.html` — hover hint button
- `src/face_recog/templates/partials/merge_row.html` — merge suggestion row
- `src/face_recog/templates/partials/merge_page.html` — page of merge rows (infinite scroll)
- `src/face_recog/templates/partials/singleton_grid.html` — singleton faces page (infinite scroll)
- `src/face_recog/templates/partials/lightbox.html` — Alpine lightbox component

### Modified files
- `src/face_recog/templates/base.html` — Tailwind + htmx + Alpine, dark nav, Google Fonts
- `src/face_recog/templates/index.html` — Tailwind dark stats cards
- `src/face_recog/templates/clusters.html` — htmx hints, Tailwind dark cards
- `src/face_recog/templates/cluster_detail.html` — Alpine selection, htmx scroll, Tailwind
- `src/face_recog/templates/singletons.html` — Alpine + htmx + infinite scroll (NEW)
- `src/face_recog/templates/persons.html` — Tailwind person cards
- `src/face_recog/templates/person_detail.html` — htmx + infinite scroll (NEW) + Tailwind
- `src/face_recog/templates/photo.html` — Tailwind face overlays
- `src/face_recog/templates/merge_suggestions.html` — htmx infinite scroll, Tailwind
- `src/face_recog/templates/find_similar.html` — Alpine selection, Tailwind
- `src/face_recog/templates/compare.html` — Alpine selection, Tailwind
- `src/face_recog/templates/search.html` — Tailwind
- `src/face_recog/static/app.js` — Alpine selection component + htmx error handler + lightbox
- `src/face_recog/static/style.css` — replaced by Tailwind output (delete old)
- `src/face_recog/app.py` — add HTML-fragment endpoints for htmx
- `src/face_recog/db.py` — add offset param to get_cluster_faces

### Unchanged
- All Python backend files except app.py and db.py
- All test files (add new tests for fragment endpoints)

---

## Implementation Tasks

### Task 1: Foundation — Tailwind CLI + base.html + Makefile

1. Download Tailwind standalone CLI for macOS ARM
2. Create `tailwind.config.js` with dark gallery theme
3. Create `src/face_recog/static/input.css` with Tailwind directives
4. Build CSS: `./tailwindcss -i ... -o ...`
5. Create `Makefile` with help, serve, scan, css, css-watch, test, lint targets
6. Rewrite `base.html`:
   - Add Google Fonts link (DM Sans)
   - Add htmx and Alpine.js CDN scripts
   - Dark navigation bar with amber logo
   - Human/Pet section toggle as pill buttons
   - Search input styled for dark theme
7. Add `[x-cloak]` and htmx indicator styles to input.css
8. Rewrite `app.js`: Alpine selection component + htmx error handler
9. Add `.gitignore` entry for `tailwindcss` binary
10. Commit

### Task 2: Lightbox component

1. Create `partials/lightbox.html` — Alpine component:
   - Full-screen black overlay
   - Displays clicked face's original photo (via `/api/photos/{id}/image`)
   - Close on click outside, Escape key, or X button
   - Fade transition
2. Include lightbox in `base.html` so it's available on every page
3. All face thumbnails get `@click` to open lightbox (via Alpine store)
4. Commit

### Task 3: Dashboard (index.html)

1. Redesign stats cards: dark cards with amber stat values
2. Large numbers with subtle animation on load
3. Action buttons styled for dark theme
4. Commit

### Task 4: Clusters list (clusters.html)

1. Create `partials/cluster_card.html` — dark card, face mosaic, hover overlay
2. Create `partials/cluster_hint.html` — amber assign button
3. Add `GET /api/clusters/{id}/hint-html` endpoint in app.py
4. Rewrite `clusters.html` — htmx hover hints, no inline JS
5. htmx assign: POST removes card from grid
6. Commit

### Task 5: Cluster detail (cluster_detail.html)

1. Create `partials/face_grid.html` for infinite scroll pages
2. Add `GET /api/clusters/{id}/faces-html` endpoint (returns HTML fragment)
3. Add `offset` param to `db.get_cluster_faces()`
4. Rewrite `cluster_detail.html`:
   - Alpine `faceSelection()` for selection state
   - htmx infinite scroll via `hx-trigger="revealed"`
   - Assign form with Alpine toggle for "Create new" input
   - Dismiss/exclude via fetch (Alpine @click) with confirm
   - Selection bar sticky, amber-themed
5. Commit

### Task 6: Singletons (singletons.html) — add infinite scroll

1. Create `partials/singleton_grid.html` for infinite scroll
2. Add `GET /api/singletons/faces-html` endpoint
3. Rewrite `singletons.html`:
   - Alpine selection
   - htmx hint quick-assign (removes face on success)
   - Infinite scroll (NEW — was capped at 200)
   - Assign/dismiss selected via Alpine @click
4. Commit

### Task 7: Merge suggestions (merge_suggestions.html)

1. Create `partials/merge_row.html` — dark card, face thumbnails, similarity badge
2. Create `partials/merge_page.html` — loop + scroll trigger
3. Add `GET /api/merge-suggestions-html` endpoint
4. Modify `POST /api/clusters/{id}/assign` and `POST /api/clusters/merge` to return empty HTML for htmx requests (detect HX-Request header)
5. Rewrite `merge_suggestions.html` — htmx loads initial page, scroll triggers more
6. ~120 lines of inline JS eliminated
7. Commit

### Task 8: Find similar + Compare

1. Rewrite `find_similar.html` — Alpine selection, Tailwind, claim via fetch
2. Rewrite `compare.html` — Alpine selection per group, swap via fetch
3. Commit

### Task 9: Person detail + Photo view

1. Rewrite `person_detail.html`:
   - Add infinite scroll for face samples (was hardcoded at 60)
   - Add `GET /api/persons/{id}/faces-html` endpoint
   - Tailwind dark styling
2. Rewrite `photo.html`:
   - Face box overlays styled for dark theme
   - Assign/unassign via htmx (no page reload)
3. Commit

### Task 10: Persons list + Search + final cleanup

1. Rewrite `persons.html` — dark person cards with face count
2. Rewrite `search.html` — dark search form and results
3. Delete old `style.css` (replaced by Tailwind output)
4. Remove any remaining inline JS
5. Run full test suite + coverage
6. Manual smoke test every page
7. Commit

---

## Key Patterns

### Alpine selection (shared across templates)

```javascript
// app.js
Alpine.data('faceSelection', () => ({
  selected: new Set(),
  get count() { return this.selected.size; },
  get hasSelection() { return this.selected.size > 0; },
  toggle(id) { this.selected.has(id) ? this.selected.delete(id) : this.selected.add(id); },
  selectAll(ids) { ids.forEach(id => this.selected.add(id)); },
  clear() { this.selected.clear(); },
  get ids() { return [...this.selected]; }
}));
```

Usage in template:
```html
<div x-data="faceSelection()">
  <div x-show="hasSelection"> ... </div>
  <div @click="toggle(123)" :class="{ 'ring-2 ring-gallery-accent': selected.has(123) }">
```

### htmx infinite scroll

```html
<!-- Last element in a page triggers loading the next -->
<div hx-get="/api/endpoint?offset=200"
     hx-trigger="revealed"
     hx-swap="outerHTML">
  Loading...
</div>
```

### htmx-aware endpoints

```python
@app.post("/api/clusters/{cluster_id}/assign")
def assign_cluster(request: Request, cluster_id: int, person_id: int = Form(...)):
    ...
    if request.headers.get("HX-Request"):
        return HTMLResponse("")  # htmx: remove the element
    return RedirectResponse(...)  # browser: redirect as before
```

### Lightbox (Alpine store)

```javascript
Alpine.store('lightbox', {
  open: false,
  photoId: null,
  show(photoId) { this.photoId = photoId; this.open = true; },
  close() { this.open = false; this.photoId = null; }
});
```

```html
<!-- On any face thumbnail -->
<img @click.stop="$store.lightbox.show({{ face.photo_id }})">
```

---

## Verification

After each task:
- `uv run pytest tests/ -v` — all green
- `make lint` — ruff + mypy clean
- `make css` — Tailwind compiles
- Manual: browse the affected pages

After all tasks:
- Browse every page in dark theme
- Test lightbox on face thumbnails
- Test infinite scroll on singletons (new) and person detail (new)
- Test selection + assign/dismiss on cluster detail and singletons
- Test merge suggestions infinite scroll + merge actions
- Check browser console: no JS errors
- Verify inline JS reduced from 386 lines to ~0 (only Alpine @click expressions)
