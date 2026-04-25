/* ===================================================================
   Ritrova bootstrap — imports the ES-module split (ADR-012 §M1 step 6).
   -------------------------------------------------------------------
   Static imports keep evaluation synchronous so Alpine's `alpine:init`
   listeners are registered BEFORE Alpine fires the event.

   Cache-busting: every import below carries the `?v=N` query so that a
   bump to N forces a fresh fetch of every sub-module. The browser
   caches sub-modules under their full URL (path + query) — without the
   query, a hard-refresh in some browsers still serves the stale copy.

   When you change ANY of the imported files, bump `?v=` HERE (in every
   import below) AND in base.html's `app.js?v=N` so the bootstrap itself
   reloads. Two places, mechanical.

   Load order matters:
     1. fetch-wrapper    installs the global fetch interceptor + htmx
                         error listeners BEFORE any other module runs
                         network requests.
     2. dialogs          exposes window.confirmDialog / claimFaces.
     3. undo             registers the HX-Trigger listener and the `z`
                         keybinding.
     4. alpine-components  registers every Alpine.store / Alpine.data on
                         'alpine:init'. Must finish before Alpine itself
                         runs, which it does because `<script type=
                         "module">` is deferred and executes before the
                         deferred Alpine CDN script.
   =================================================================== */

import './js/fetch-wrapper.js?v=22';
import './js/dialogs.js?v=22';
import './js/undo.js?v=22';
import './js/alpine-components.js?v=22';
