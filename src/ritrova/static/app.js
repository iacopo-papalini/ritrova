/* ===================================================================
   Ritrova bootstrap — imports the ES-module split (ADR-012 §M1 step 6).
   -------------------------------------------------------------------
   Load order matters:
     1. fetch-wrapper    installs the global fetch interceptor + htmx
                         error listeners BEFORE any other module runs
                         network requests.
     2. dialogs          exposes window.confirmDialog / claimFaces.
     3. undo             registers the HX-Trigger listener and the `z`
                         keybinding; keeps the undo state module-local
                         and exports the two hooks alpine-components
                         needs.
     4. alpine-components  registers every Alpine.store / Alpine.data on
                         'alpine:init'. Must be imported before Alpine
                         itself runs, but since Alpine is loaded with
                         `defer` after this module, ordering is safe.

   Served with type="module" in base.html — bumping ?v= invalidates the
   browser's cached import graph.
   =================================================================== */

import './js/fetch-wrapper.js';
import './js/dialogs.js';
import './js/undo.js';
import './js/alpine-components.js';
