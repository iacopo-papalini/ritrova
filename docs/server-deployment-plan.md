# Server Deployment Plan

Living plan for taking Ritrova from a hobby project on a MacBook to a family service on
the home server. Updated during work — tick items, don't delete them.

**Started:** 2026-10-24 · **Rewritten:** 2026-10-25 around the Postgres decision

## What this document owns

This plan is **temporary and forward-looking**: intent, sequencing, decisions and their
reasoning. It ends when the work lands.

`core-knowledge/07-operations/` is **durable and present-state**: what is true of the
running system.

As each workstream lands, its facts become present-state and belong in the chapter; this
file is then closed and archived, not maintained alongside it.

## Deployment: deferred, then resumed (2026-10-25)

**Resumed the same day.** The local migration and cutover are done, and the user has put
paiolo back in scope with four parts: a Dockerfile running Ritrova and Postgres together,
restoring the dump into paiolo's Postgres, a smoke test on the box, and the service
accepting connections from anywhere on the local network. `ritrova-67` is implementing,
in a worktree off main.

The LAN exposure is a decision already taken — *"I'm aware of the security implications"*
— and is not to be re-litigated. Note that the **W5-before-W6 rule is untouched** by it:
that rule governs `photos.papalini.casa` resolving on the public internet, not the house
network, and still binds if public exposure is added.

Main is at `170ddb7`: eight Postgres commits, linear history, 434 tests green. Nothing
pushed.

> **The Postgres data is no longer reproducible from `faces.db`.** Two hand-run cleanups
> live only in the database: **991 Windows-path duplicate sources deleted**, and **30
> faces relabelled from Lilia to Carla Gabbrielli**. Both conditions still exist in the
> SQLite source, so a fresh migration silently reintroduces them and nothing fails.
>
> The SQL is not in the repo. That makes
> `~/Progettini/ritrova-dumps/ritrova-20261025-2022.dump` (155 MB, custom format) the only
> artifact carrying that work, and it sits outside the repo and outside any backup. Until
> the cleanup is captured as a migration or a script, the pipeline is not reproducible —
> there is one good dump instead.
>
> Current contents: 32,721 sources · 67,460 findings · 33,751 named faces.

### Superseded: deployment deferred (earlier on 2026-10-25)

**paiolo is out of scope for now. Everything stays local, the cutover included.** Relayed
from the user via `ritrova-67`: *"paiolo is not going to be involved yet - we will work in
local - even the migration to postgres will"*.

What this means for the decisions already taken, none of which are withdrawn:

- **D1** keeps its substance — Postgres replaces SQLite — but its second clause, Postgres
  *on paiolo* with both machines connecting over the LAN, is deferred. One machine, one
  local cluster.
- **D8** (no Ritrova on paiolo until Postgres is there) is **moot, not satisfied**. Do not
  later read it as completed: nothing was deployed, the question simply stopped applying.
- **D12** (`--locale=C.UTF-8`) still binds whenever a paiolo cluster is eventually created,
  and still cannot be changed afterwards.
- **D15** (`data_dir`) still binds; the paiolo-volume gap against W2.2 is deferred with the
  rest, not resolved.
- **W2–W7 are all deferred.** They are kept below because the analysis holds, not because
  they are queued.

Nothing in the branch needed changing: `compose.dev.yaml` binds `127.0.0.1:5432` rather
than `0.0.0.0`, the default DSN is `127.0.0.1`, and there is no reference to paiolo in any
code, compose file or config.

## Sessions and ownership (2026-10-25)

Four Claude sessions work this repo. The split is by *machine and surface*, not by task,
so that two of them never hold the same file.

| Session | Owns | Scope |
|---|---|---|
| `ritrova-f4` | this plan | Director. Decisions, sequencing, briefing. Writes no production code. |
| `ritrova-67` | the migration | W1.2–W1.8, on `.claude/worktrees/w1_postgres_async` off main. |
| `ritrova-11` | paiolo's Nextcloud sync | Scoped by its own user to the sync, not to this plan. It does not take workstreams from the director; W2 work is offered to its user, not assigned. |
| `ritrova-3b` | `core-knowledge/` | Kipi. The durable KB this plan eventually folds into. |

One thing stays out of every worker's hands and needs the user directly: **W1.9 cutover**,
the point at which the SQLite file stops being the live copy. W1.6 is non-destructive and
delegated outright.

No session's message is another session's authorization. A worker blocked on a permission
prompt takes it to the user, not to the director.

## Topology (verified 2026-10-24)

```
Internet  2.39.73.4
   │ router forwards 80/443
   ▼
Synology DS220+   172.20.42.2   DSM 7.3.2
   nginx reverse proxy + Let's Encrypt
   ├─ cloud.papalini.casa:443  → http://localhost:8080    Nextcloud SERVER
   ├─ wiki.papalini.casa:443   → http://localhost:18080
   └─ photos.papalini.casa:443 → http://172.20.42.3:8787  ← to add
          │
          ▼
"paiolo"  172.20.42.3   Ubuntu 22.04.5, Celeron J4125 (4c/4t), 7.6 GB RAM
   /srv/family-data  938 G volume, 166 G used / 725 G free
   ├─ docker (snap): pihole — owns :80 and :53 for the whole house
   ├─ nextcloudcmd sync CLIENT → /srv/family-data/Nextcloud (~107 K photos)
   ├─ postgres container                                   ← to add
   └─ ritrova container :8787                              ← to add

MacBook (Apple Silicon)   ~/nextCloud/Photos   scan + caption extras → analyse
   connects to paiolo's Postgres over the LAN
```

Three facts that shape everything below:

- **Nextcloud's server is on the NAS, not on paiolo.** paiolo runs the desktop sync
  client, so it holds a full local copy of the photo tree. Serving reads hit local disk.
- **TLS terminates on the NAS.** paiolo has nothing on :443 and pihole owns :80.
- **paiolo is also the household's DNS server.** Anything that starves it breaks the whole
  house, not just photos.

## Decisions taken

| # | Decision | Date |
|---|---|---|
| D1 | **Postgres replaces SQLite.** Running locally on the Mac: `compose.dev.yaml` binds `127.0.0.1:5432` and the default DSN is loopback. The paiolo half — a server instance with both machines connecting over the LAN — is in progress, not fact. | 2026-10-25 |
| D2 | **The app becomes async**: FastAPI + asyncpg on a single-threaded event loop. No shared connection, no `RLock`, no request threadpool. | 2026-10-25 |
| D3 | **Image work runs on a bounded thread executor**, sized to core count, with a semaphore limiting concurrent decodes. | 2026-10-25 |
| D4 | **Alembic owns the schema**, with hand-written migrations. No ORM and no SQLAlchemy in the query layer — SQLAlchemy is present only as Alembic's dependency. | 2026-10-25 |
| ~~D5~~ | **Deferred 2026-10-25 — not a decision.** Postgres backups are deliberately out of scope until the migration lands. The material below is retained as input for that conversation, not as an agreed design. The user has not approved an approach. | — |
| D6 | Google OIDC per `docs/auth-session-design-home-server.md`. No new SSO box, and **no offline fallback door**. | 2026-10-24, confirmed 2026-10-25 |
| D7 | The server image is serve-only — no model runtimes. Manual-finding creation (Shift+drag) is disabled there and stays a Mac-side operation. | 2026-10-24, confirmed 2026-10-25 |
| D8 | **Ritrova does not run on paiolo again until Postgres is there.** No SQLite serving on the server, not even read-only. | 2026-10-25 |
| D9 | **Dates get three different types, not one.** `timestamptz` for the five clean audit columns; `timestamp` (no zone) for `sources.taken_at`; `text` for all four `source_path_metadata` date columns. See *Why D9*. | 2026-10-25 |
| D10 | The Postgres DSN is **`RITROVA_DSN`**, a separate variable from `FACE_DB`, so both coexist during the port and W1.6's reader keeps its SQLite path. | 2026-10-25 |
| D11 | **After cutover the filesystem database is deleted.** `faces.db` and its sidecars leave the Nextcloud tree entirely. Retained snapshots are the rollback, not the live file. | 2026-10-25 |
| D12 | **Every Postgres cluster is initdb'd `--locale=C.UTF-8`.** Not the image default. Irreversible after creation. Applies to paiolo as much as to the Mac. | 2026-10-25 |
| D13 | **UUIDv7 primary keys on all six surrogate-keyed tables**, replacing integer identities. User's decision, against `ritrova-67`'s recommendation. | 2026-10-25 |
| D14 | **Strong typing at the boundary.** Dataclasses carry real `datetime`, not the strings they hold today. | 2026-10-25 |
| D15 | **`data_dir` becomes an explicit setting**, alongside `RITROVA_DSN`. `db_path.parent` was load-bearing and there is no longer a db path. | 2026-10-25 |

Superseded on 2026-10-25, kept for the reasoning: the server-owns-SQLite split, the
scan-delta handoff, the Mac-as-authoritative-copy rule and the snapshot-as-reference rule.
All of them existed to work around one SQLite file being reachable by two machines.
Postgres removes the problem rather than managing it.

### Why D1

`FACE_DB` points inside the Nextcloud sync folder on both machines. A live SQLite database
and its `-wal` / `-shm` sidecars are ordinary files to a sync client, shipped whole and
independently. Two writers means conflict copies and eventually a corrupt archive of
irreplaceable curation work. Every alternative considered — a delta handoff, or a
maintenance-mode cycle with the whole 486 MB file moving each way — was a protocol layered
over a database that cannot be shared. Postgres is shareable by design.

### Why D2

The current design is one SQLite connection behind one `threading.RLock`, taken by every
database method. A single grid page fires ~50 thumbnail requests that all queue on it —
so the serialisation bites with one user, not just several. A per-thread connection pool
would have fixed that for about a day's work, but Postgres plus an event loop removes the
lock, the threadpool and the GIL contention together, and it is the natural pairing with
D1.

### Why D3

A single-threaded event loop and CPU-bound work are a bad combination: one 300 ms JPEG
decode stalls every other request on the loop, including the other 49 thumbnails. Pillow
releases the GIL inside its C codec loops, so a *thread* executor gives real parallelism
across the four cores without pickling image bytes between processes. Threads, not
processes, and bounded — anyio's default of 40 would thrash a 4-core box.

### Why D15

`FaceDB` took a `db_path`, and `db_path.parent` quietly resolved three things: video
frames under `tmp/`, thumbnails, and analysis reports. With no SQLite file there is no
parent, so `FaceDB.connect()` now takes an explicit `data_dir`.

This is a deployment setting, not an implementation detail. On the Mac it is the existing
`ritrova_data_win/`. **On paiolo it must be a writable volume, which W2.2 does not
currently provide for** — the failure mode otherwise is a container that starts, serves
pages, and cannot write a thumbnail. Flagged by `ritrova-67` before the fact rather than
after.

### Why D13 and D14

Both taken by the user directly with `ritrova-67` on 2026-10-25, the director not in the
loop. Recorded here as settled, not as proposals.

**D13 — UUIDv7.** `ritrova-67` recommended against it (W1.6 loses id-equality as a
verification, every route and template changes, wider indexes on a 4-core box) and asked
explicitly. The user chose it anyway, and confirmed the Mac will always be connected — so
no offline-merge requirement is driving it. Their reasoning: *"UUIDv7 keep time-sorting
order, stronger than int."* It also matches the standing CLAUDE.md rule that UUIDs are
always v7.

It cost less than predicted. **Only six tables have a surrogate key** — `sources`,
`scans`, `subjects`, `findings`, `descriptions`, `circles` — and the rest are keyed by
their foreign keys, so they came along free. `cluster_findings.cluster_id` stays
`integer`: a cluster is a label the clustering algorithm emits, not a row in any table,
and converting it would have been an invisible mess.

The ids carry **real time, not migration time** — each row's v7 timestamp comes from its
own creation column (`detected_at`, `created_at`, `scanned_at`), and for `sources`, which
has none, from the earliest scan of that source. Stamping all 302,612 with the migration's
wall clock would have been shorter and would have discarded the exact property the
decision was made for.

**D14 — strong typing.** User: *"I prefer strong typing — my feeling is that postgres is
much more typed-coding friendly."* Larger than it sounds: `source.taken_at[:10]` is sliced
in two templates and parsed with `int(taken_at[:4])` in `clusters.py`, all of which break
the moment the field stops being a string. Handled inside W1.3+W1.4.

### Why D12

Found by `ritrova-67` while standing up the development container, and it is the kind of
thing that is invisible until it is expensive.

SQLite's default collation is `BINARY` — byte order. A Postgres cluster initialised with
the image's default `en_US.UTF-8` collates differently, so **every `ORDER BY name` in the
application silently re-sorts**: accented Italian names move, case ordering changes.
Nothing errors. Nothing fails a test. The lists simply come back in an order the user did
not have yesterday, and the cause is three layers below anywhere anyone would look.

`C.UTF-8` reproduces today's ordering exactly, and is identical on macOS and on paiolo, so
development and production agree rather than diverging in a way only production shows.

**This cannot be changed after the cluster is created** — it requires a dump and restore.
It is therefore a precondition on creating paiolo's volume, not a tuning step afterwards,
and it belongs in W2's first action rather than its checklist.

### Why D11

The user's decision, 2026-10-25. It is the right end state — one database, in one place,
with one access path — and it closes the problem this whole plan started from: a live
SQLite file sitting inside a sync tree, reachable by two machines.

It also collapses most of W0. **W0.2 (the sync exclusion) becomes unnecessary** — there is
no point engineering a careful exclusion for a file that is about to be deleted, so that
item is dropped rather than done. **W0.4 (the NAS's stale copy)** resolves the same way:
the deletion propagates.

One consequence that must not be lost (the second, about backups, is parked):

- **The snapshots become the only SQLite that exists.**
  `data/snapshots/faces-20261024-verified.db` and a final pre-cutover snapshot are the
  rollback path, and they are *outside* the Nextcloud tree and gitignored — which means
  nothing is backing them up. They need a deliberate home before the delete, not after.
- **D5 stops being prudence and becomes the only backup there is.** While `faces.db`
  exists in the sync tree, the NAS holds a copy of the curation by accident. After D11 it
  does not. Every copy of 70,002 findings is then Postgres plus whatever `pg_dump`
  successfully ships — over a sync that has not yet exited `rc=0` once.

Sequencing that falls out of this: **do not delete the filesystem database until a
`pg_dump` has been written, shipped and restored at least once.** Not written — restored.

### Why D9

`ritrova-67` profiled the real 486 MB file on 2026-10-25. A single uniform
`TEXT` → `timestamptz` rule — which both of us had assumed — corrupts two of the three
kinds of date column in this database.

**1. `sources.taken_at` is EXIF colon format, never ISO.** `2002:08:09 20:07:38`. 30,863
rows, 2,822 NULL, and 27 rows of the `0000:00:00 00:00:00` sentinel. It is naive local
camera time carrying no zone at all. A `timestamptz` cast fails outright on the sentinels
and invents a timezone for everything else. It becomes **`timestamp` without zone**, with
the 27 sentinels converted to NULL — honest about what it is, still sortable, and
arithmetic still works. Note also that per the project's own rule this column is *not* the
authoritative date: paths are.

**2. `scans.scanned_at` is mixed within one column.** 68,168 rows are aware ISO with
`+00:00`; 9,809 rows are naive and space-separated (`2026-04-30 20:11:22`), left by a
legacy backfill. Parsed naively in a Europe/Rome session those 9,809 shift by two hours.
The migration must assume UTC explicitly for the naive ones rather than let the session
timezone decide. `findings.detected_at`, `subjects.created_at`,
`finding_assignment.curated_at` and `source_path_metadata.indexed_at` are uniformly aware
and are the only clean `timestamptz` candidates.

**3b.** `scan_types.introduced_at` and `retired_at` stay `text` for the same reason —
they hold partial dates like `'2026-04'`. Not covered when D9 was written; added
2026-10-25.

**3. The four `source_path_metadata` date columns are variable-precision partial dates**,
and must stay `text`. `1994`, `2002-08`, `2002-08-09` — 155 year-only, 4,024 month-only,
29,530 full, 3 unknown, which is exactly what the sibling `date_precision` column exists
to record. `source_search.py` range-filters them with string `>=` / `<=` against
`idx_source_path_metadata_date`, which works *because* they are lexicographic ISO
prefixes. Casting them to `date` would destroy the precision model and the index strategy
in one move. (`filename_date` also reaches `2055`, so there is genuine garbage in there.)

### Note on D6

The design doc's "Google login in testing mode" assumption is more pessimistic than it
needs to be: an app requesting only `openid`, `email` and `profile` is exempt from app
verification, is not capped to the trusted-tester list, and its authorizations do not
expire after 7 days. Recorded with sources in the **Google Project Configuration** section
of `docs/auth-session-design-home-server.md`.

Reviewed 2026-10-25 and kept as the single way in. The two objections both fall:

- *Everyone needs a Google account.* The family is on Android; the accounts already exist.
- *A dead WAN locks the house out of its own photos.* It does not. The photo tree is a
  Nextcloud share on the NAS and stays readable by its own path. Ritrova is the curation
  and search layer, not the only way to see the pictures — so an outage costs features,
  not access.

Nothing to build: no device tokens, no LAN bypass, no second credential store. Each of
those would have been an unauthenticated path into a write-capable UI, which is exactly
what the W5-before-W6 rule exists to prevent.

### Note on D7

Reviewed 2026-10-25, accepted unchanged. D1 improves it without changing it: with both
machines on the same Postgres, a finding drawn on the Mac is visible on the server at
once, so the feature is not absent from the deployment — it has a location. The residual
cost is that the Mac becomes a required participant for that one operation, and the Mac
is a laptop that leaves the house.

## Sequencing rule

**W6 (public HTTPS) must not land before W5 (auth).** The moment `photos.papalini.casa`
resolves, an unauthenticated write-capable UI is on the internet. Until then, 8787 is
firewalled to the NAS only.

- **Phase 0 — stop the bleeding:** W0. Small, and worth doing before a migration that will
  take weeks.
- **Phase 1 — the migration:** W1. Done on the Mac, against a local Postgres.
- **Phase 2 — on the box:** W2, W3.
- **Phase 3 — make it usable:** W4.
- **Phase 4 — expose it:** W5, then W6.
- **Phase 5 — sustainable:** W7.

---

## W0 — Safety now

The SQLite database stays live and in the sync tree until W1 completes. That is weeks, so
these are worth doing regardless.

D8 shrinks this phase considerably. The Mac is the only process that ever opens the
database, so the whole concurrent-writer class of risk is gone: no two clients racing a
write, no half-synced `-wal` being opened by a second reader. What remains is narrower —
Nextcloud may upload a 486 MB file while SQLite is mid-transaction, which costs a usable
*backup*, not a usable *database*. The live file on the Mac is never at risk from the
sync; only the copy of it is.

> **Parked: backup design.** Deferred by the user on 2026-10-25 as premature — it is a
> real problem and will be addressed, but not before the migration lands. Nothing below
> is decided. It is kept only so the analysis does not have to be redone, and it must not
> be read as an agreed plan. Several of these points came from peer sessions and were
> written up by the director without the user approving the approach, which was the
> error.
>
> **How D5's health is measured.** The four non-zero exit codes seen on 2026-10-25
> (`143, 1, 124, 143`) were diagnosed by `ritrova-11` and are not evidence of a bad
> transport: two were its own deliberate SIGTERMs, one was a Qt6 TLS plugin missing after
> a client upgrade (fixed by `qt6-qpa-plugins`, now `OpenSSL 3.0.2`), and one was a 4h
> timeout firing on a first full reconciliation with no errors in the log — healthy but
> slow. The timeout is now 12h.
>
> The real finding is worse than a bad exit code and is a design input, not an incident:
> **the flock skip exits 0 by design.** "Skipped because the previous run is still going"
> is indistinguishable from success to anything watching exit status. If a reconciliation
> ever outlasts 24h, every nightly backup queues behind the lock and is silently skipped
> while the sync continues to look healthy. For an off-machine backup that is the worst
> available failure mode — and it is exactly how the Qt6 TLS gap, or the dead cron before
> it, could have suppressed backups for weeks unnoticed.
>
> Therefore D5 alarms on **the age of the newest dump as seen from the NAS side** — older
> than ~48h is a failure — and never on the sync's return code. Unproven until an `rc=0`
> exists; `ritrova-11` is watching for one.
>
> **The alarm does not run on paiolo.** Under D11 that one box holds Postgres, runs the
> sync client and writes the dumps, so a dead disk takes the live database and every
> local dump in the same instant — which is the exact scenario D5 exists for, and the
> exact scenario a check hosted on that box cannot report. A paiolo-side check covers the
> silent-skip case and misses the one that destroys data.
>
> It runs on **the Mac**, which already syncs the same tree and therefore sees dumps
> arrive *from* the NAS rather than inspecting where they were written. That answers the
> stronger question — has a dump from the last ~48h actually left paiolo and come back
> down to a second machine — and it is the only check that passes only if the whole round
> trip works. (`ritrova-11`, 2026-10-25.)
>
> Caveat the director adds: the Mac is a laptop that sleeps and travels, so the check will
> miss its schedule routinely and must tolerate that — alarm on dump age, never on "the
> check did not run at the expected minute", or it will cry wolf until it is ignored.
>
> **Retention inside a synced directory needs care.** 14 daily plus 8 weekly means routine
> deletions in a folder Nextcloud watches. Benign in steady state, but a future client
> upgrade that resets the journal again will treat locally-present, remotely-deleted dumps
> as new content and re-upload them — exactly what is happening to `ricorda_data` right
> now. Retention silently undoes itself and disk creeps. Enforce retention **by age on
> both sides** rather than by deletion on one, and never let `.sync-exclude.lst` cover the
> dump directory.

> **A read-only open is not a no-op.** Opening a WAL-mode SQLite database with `mode=ro`
> still **creates** `-wal` and `-shm` when they are absent. `ritrova-67` put a 0-byte
> `faces.db-wal` and a 32 KB `faces.db-shm` into `ritrova_data_win/` at 11:29 on
> 2026-10-25 purely by profiling the file. No data changed — `faces.db` is byte-identical
> at 485,953,536 bytes and keeps its `24 ott 17:28` mtime, and a zero-length `-wal` means
> nothing was ever written through it. The sidecars were left in place, since the user has
> said twice not to touch that directory.
>
> This extends D1's reasoning rather than merely being a slip: the plan frames the hazard
> as a live database being *written* inside a sync tree, but a well-meant read-only
> inspection from any session adds two files to the sync set. The 30 April backup has
> carried its own sidecars since then, so this has happened before and the sync tolerated
> it. All further work reads the scratchpad copy.

- [x] **W0.1 — One-off safety snapshot** (2026-10-24).
      `data/snapshots/faces-20261024-verified.db`, outside the Nextcloud tree and
      gitignored. `quick_check: ok`, 33,712 / 70,002 / 77,977 / 154, newest scan
      `2026-10-18T16:44:04Z`.
- [ ] **W0.2 — Exclude `tmp/` and `yolo11m.pt` from the sync.** Narrowed 2026-10-25.
      Stands on its own merits — 3.8 GB of cache re-walked every run — independent of any
      backup design, which is parked.

      The database half is dropped under D11 — no point engineering an exclusion for a
      file about to be deleted. The cache half is the opposite of tidiness: `tmp/` is
      3.8 GB of thumbnails and video frames re-walked on every single run, forever. Every
      minute of that is a minute a nightly `pg_dump` can spend queued behind a
      reconciliation. Excluding it is what keeps D5's backups from ever being skipped.
- [~] **W0.3 — Delete the resurrected `ricorda_data` / `ricorda_data_win`.** Done on the
      **Mac** 2026-10-25 by `ritrova-11`, on the user's instruction, and propagating to
      paiolo through the sync rather than by touching the server. All 30 files in
      `ricorda_data` were byte-identical duplicates of files already in `ritrova_data`;
      `ricorda_data_win` was empty.

      **Open race.** paiolo's sync journal was reset by the client upgrade, so it reads
      those directories as new local content and is uploading them. If the upload wins,
      the trees come back and the deletion has to be repeated on the Mac. Watch one full
      run before calling this closed.
- [–] **W0.4 — The NAS's stale copy.** **Resolved by D11** — the deletion propagates.
- [ ] **W0.6 — Give the snapshots a backed-up home.** Under D11 they become the only
      SQLite copies in existence, and today they live in a gitignored directory that
      nothing replicates. Must land *before* the cutover delete, not after.
- [ ] **W0.5 — paiolo's `FACE_DB` points at nothing.** Two in-tree `faces.db` copies were
      moved to `/home/iacopo/ritrova-archive/` on 2026-10-25; both are intact, nothing was
      deleted. `/home/iacopo/ritrova/.env` still names the old path. Inert under D8, since
      nothing on paiolo opens it — but the plan must not assume that variable resolves.
      Fixed as part of deployment, not before.

## W1 — Postgres and async migration

The large one. Everything else waits behind it.

- [x] **W1.1 — Postgres on the Mac** (2026-10-25). `postgres:17` (17.11) via
      `compose.dev.yaml` in the worktree, bound to `127.0.0.1:5432` only, named volume,
      `pg_isready` healthcheck, initdb `--locale=C.UTF-8` per D12. Original item: A local development instance, and the one
      the real data lands in. **Not on paiolo.** The server's database is created at
      deployment time by restoring a dump of this one (W2), not built empty and migrated
      into. Decided 2026-10-25: code first, data second, deployment third.

      Sizing for paiolo, when W2 gets there, corrected 2026-10-25 by `ritrova-11` from
      measurement: the box has **7.6 GB RAM**, so the earlier 300–400 MB budget was the
      wrong shape of number. `shared_buffers` of 256–512 MB is right, but the figure that
      governs behaviour is `effective_cache_size` — what makes serving 156 GB of photos
      from a Celeron tolerable is the page cache, and Postgres must be told the truth
      about it rather than starved. pihole is untouchable: it is DNS for the whole house.
- [x] **W1.2 — Alembic baseline** (2026-10-25). Revision `a9b07b12530f`, async template,
      `env.py` reads `RITROVA_DSN` and rewrites to `postgresql+asyncpg://` itself.
      `target_metadata = None`, raw DDL via `op.execute` per D4. 12 tables, 14 indexes,
      matching SQLite one for one; `downgrade base` → `upgrade head` round-trips clean.
      Embedding round-trip bit-exact. `'Marco' LIKE 'marco%'` confirmed false on the real
      server, so the four `ILIKE` sites are measured, not predicted. Identities are
      `GENERATED BY DEFAULT` so W1.6 can preserve every primary key. Superseded item: An initial migration reproducing today's schema exactly,
      hand-written. Replaces schema-creation-on-connection-open, and makes W1.9's
      correctness checkable. Supersedes the old schema-version-stamp item.
- [x] **W1.2b — Duplicate index on `sources.file_path` dropped** (2026-10-25, `a0fb7f9`).
      Migration `b2c41f08ad77` revising the baseline, so the already-migrated cluster
      reaches the same schema by the same path as a fresh one; downgrade recreates the
      index and the round trip is tested. The redundancy was **verified, not assumed** —
      `EXPLAIN` gives an identical plan with and without `idx_sources_path`, both for the
      equality lookup (which uses `sources_file_path_key`) and for the
      `LIKE 'prefix%' ORDER BY file_path` scan, which seq-scans either way. Pure write
      cost. Original item: The column carries both a UNIQUE constraint and
      `idx_sources_path`; the baseline reproduces it deliberately, because a baseline that
      quietly improves on its source is no longer a baseline you can verify against.
      Found by `ritrova-67`, 2026-10-25.
- [x] **W1.3 + W1.4 — `db/` on asyncpg and the whole app async. Done 2026-10-25.** The
      app runs end-to-end on Postgres: 49 files, +2214/-2433, all 11 routers, 5 services,
      `undo.py`, `cluster.py`, `hints.py`, `describer.py`, `analysis.py`,
      `analysis_report.py`, `scanner.py`, 19 CLI commands, 3 templates and the frontend
      JS. Whole tree typechecks, all three pre-commit hooks pass, reads and writes
      verified against the migrated 302,612 rows. Claim→undo and dismiss→undo both return
      the database exactly to its prior state.

      > **The UUID switch broke things no type checker could see.** `parseInt` on a v7
      > uuid does not fail — `parseInt("01a172b4-…", 10)` returns **1**, parsing the
      > leading `01` and stopping. Seven JS sites: every finding tile would have collapsed
      > onto id 1 and drag-select would have silently selected the wrong faces.
      > `Number()` gives NaN, which then fails `Set.has` because NaN !== NaN. Three
      > server-side id parsers filtered on `.isdigit()`, so with UUIDs every entry is
      > dropped and the filter silently matches everything instead of erroring. And
      > `f"{uuid:>6}"` raises TypeError, because UUID has no `__format__`.
      >
      > Two more found only by **running** the app, not by mypy: every JSON endpoint would
      > have 500'd, since `json.dumps` serialises neither UUID nor datetime (now routed
      > through `jsonable_encoder` in `app/responses.py`); and `analyse` persisted from
      > inside its `ThreadPoolExecutor` workers, where a coroutine cannot be awaited —
      > that path could not have worked at all. Split into a sync CPU half and an async
      > `_persist_one` on the loop behind a semaphore, which made the old
      > `threading.Lock` genuinely dead.

      > **Known coupling, self-expiring.** `get_exif_date` now returns `datetime` and must
      > agree with `exif_to_naive` in the migration script, or migrated and
      > freshly-scanned rows disagree about the same photo. Two parsers held in step by a
      > comment is fragile — but only until cutover, after which the migration script is
      > run-once and dead. Do not let it outlive W1.9.

      Earlier detail, W1.3 (2026-10-25): all 18 modules on asyncpg,
      ruff and mypy clean, and **exact parity against the real 302,612 rows** — every
      aggregate equal to SQLite's answer, whole sweep in 0.26s. `search_subjects("MAR")`
      and `("mar")` now both return the same 12 people; ported literally the uppercase
      query returned nothing.

      > **The `?` placeholders were not renumbered by hand.** Queries are composed from
      > fragments (`species_filter`, `_dim_filter`, `_together_query`, a dozen
      > `",".join("?" * n)` IN-lists) and a positional `$n` must know its index where it
      > is written, which a fragment cannot. So the mixins still read `?` and one cached
      > `to_pg` rewrites them at the asyncpg boundary — one tested place instead of 242
      > chances to misnumber. **`db/` therefore still looks like SQLite until you find
      > `to_pg`.** Most IN-lists are gone regardless: `= ANY($1)` takes the list as one
      > parameter, which also removed the 500-id chunking in `delete_orphans` that existed
      > only for SQLite's variable limit.

      > **`INSERT OR REPLACE` was quietly destructive.** It deletes and re-inserts, so it
      > fires `ON DELETE CASCADE` on dependants; `ON CONFLICT … DO UPDATE` edits in place.
      > On `finding_assignment` the old behaviour could take dependent rows with it. All
      > 13 sites converted. This is a latent data-loss bug in today's SQLite app, fixed as
      > a side effect rather than found by anyone looking for it.

      Other silent-breakage sites closed while porting: `GROUP_CONCAT(f.id || ':' || …)`
      → `string_agg` with an explicit `::text`, since Postgres will not concatenate a
      uuid (and the Python side now parses `UUID(…)`, not `int(…)`); `_photo_month_index`
      sliced `taken_at[:4]`/`[5:7]` and now uses `.year`/`.month` under D14;
      `_parse_exif_date`'s `raw.replace(":", "-", 2)` deleted outright; unaliased derived
      tables in `get_random_avatars` and `count_sources_with_all_subjects`, legal in
      SQLite and not in Postgres; and `ORDER BY src.taken_at DESC` → `DESC NULLS LAST`,
      which is what SQLite was already doing for those 2,822 nulls. `recreate_subject`'s
      docstring justified id reuse with "AUTOINCREMENT only ever advances" — void under
      D13, and replaced with the better reason that a UUID is never reissued.

      Original scope: Merged 2026-10-25 by the user, after `ritrova-67`
      showed the original split could not hold: porting `db/` makes all 105 methods
      `async def`, and every caller in the routers, `services_domain/` and the CLI is
      synchronous. The app breaks at the *start* of the port, not the end, and keeping
      `_locked` does not help — a `threading.RLock` around an asyncpg call protects
      nothing. The branch is non-working mid-flight and tests are red until W1.8; both are
      expected. Reporting is per-module, 18 of them.

      Original W1.3 scope: 3,730 lines across 18 mixin modules, and ~54
      SQLite-specific constructs:

      | Construct | Count | Becomes |
      |---|---|---|
      | `?` placeholders | **242** | `$n` — mechanical, but the bulk of the diff |
      | `@_locked` methods | **105** | gone (D2) — the real measure of this workstream |
      | `SELECT EXISTS(…)` | 18 | 0/1 → real `boolean` at every call site |
      | `INSERT OR REPLACE` | 13 | `ON CONFLICT … DO UPDATE` |
      | `executemany` | 8 | asyncpg `executemany` |
      | `AUTOINCREMENT` | 8 | `GENERATED … AS IDENTITY` |
      | `LIKE` | 7 (4 user-facing) | **`ILIKE`** — see below |
      | `cur.lastrowid` | 7 | `RETURNING id` |
      | `PRAGMA` | 5 | gone |
      | `INSERT OR IGNORE` | 3 | `ON CONFLICT … DO NOTHING` |
      | `executescript` rebuilds | 3 | Alembic migrations |
      | `cur.rowcount` | 3 | asyncpg returns `"UPDATE 3"`, a status string, not an int |
      | `COLLATE NOCASE` | 3 | `ORDER BY lower(…)` |
      | `GLOB` | 1 (`scans.py:87`) | `LIKE` / `~` |

      Counts verified against the code by `ritrova-67`, 2026-10-25, correcting the
      director's grep: 4 of the 9 `PRAGMA` and 3 of the 4 `GLOB` were prose in docstrings
      and help text. Also: embeddings `BLOB` → `bytea`, and dates per **D9**.

      **Three silent-breakage sites**, none of them syntax errors, all of them wrong
      behaviour that ports cleanly and fails quietly:

      - **`LIKE` is case-insensitive for ASCII in SQLite and case-sensitive in Postgres.**
        Ported literally, the subject typeahead stops matching `marco` against `Marco`.
        Four user-facing sites: `subjects.py:296,307`, `descriptions.py:73,76`,
        `source_search.py:51`. They want `ILIKE`.
      - **NULL ordering flips.** SQLite sorts NULLs first ascending, Postgres last.
        `source_search.py:87` already guards with an explicit `CASE WHEN … IS NULL`;
        **`subjects.py:236` (`ORDER BY src.taken_at DESC`) does not**, and `taken_at` has
        2,822 NULLs — that grid silently reorders.
      - **`sqlite_sequence` has no Postgres equivalent.** After W1.6's bulk load every
        identity sequence must be `setval`'d or the first insert collides on every table.
        This is the single most likely way the migration looks successful and breaks an
        hour later.
- [ ] **W1.4 — Make the services and routers async.** Every handler becomes `async def`
      and every database call `await`. This is the whole app, not just `db/`.
- [x] **W1.6 — One-shot data migration: script written and dry-run clean** (2026-10-25).
      `scripts/migrate_sqlite_to_postgres.py`. Dry run by default, `--apply` commits, the
      whole load in one transaction that rolls back on any verification failure — a bad
      run leaves an empty database, never a half-populated one. **302,612 rows in five
      seconds.** Verifies row counts three ways (SQLite, Postgres, loader), embedding
      round-trip on 500 findings by byte equality *and* float32 equality *and*
      `octet_length/4 == embedding_dim`, every sequence value, the `finding_assignment`
      XOR invariant on loaded rows, and no `taken_at` below 1900 (how a leaked EXIF
      sentinel would show). Refuses to run against a path containing `nextCloud`, and
      refuses to load into a non-empty `sources`. **Not run with `--apply`** — that needs
      the user.

      > **SUPERSEDED by D13 — there are no sequences any more.** Kept as reasoning, not
      > as work: do not go looking for `setval` code, there is none.
      >
      > **Seed sequences from `sqlite_sequence`, never from `MAX(id)`.** They disagree on
      > every table, and not because the app is old — it was written in 2026. It is churn:
      > `analyse --force` re-scans and `scans prune` delete and reissue heavily, and
      > `descriptions` was emptied by the VLM retirement (ADR-011). `findings` has issued
      > 152,518 ids to hold 70,002 rows, so **more than half of everything ever detected
      > has since been deleted**. `sources` holds 33,712 rows with its sequence
      > at 48,571, and `descriptions` is **empty with its sequence at 22,324**. The obvious implementation — seed from the data — hands out
      > ids that have already been used, and for `descriptions` restarts from 1 against an
      > FK'd table. Nothing fails at migration time; it surfaces weeks later as rows
      > attached to the wrong parent. Found by `ritrova-67`, 2026-10-25.

      > **The UUIDv7 millisecond-overflow path is live on this data, not theoretical.** A
      > v7 counter holds 4,096 ids per millisecond, and the 9,809 legacy-backfill scans
      > all share a single timestamp — so the spill runs on every migration of this
      > dataset. `ritrova-67`'s first implementation reset to the next row's own
      > millisecond on overflow and therefore **handed out descending ids**, destroying
      > the time-ordering that D13 exists for, silently. Fixed to a strictly monotonic
      > (ms, counter). Caught by the verification, not by a test.

- [x] **W1.5 — Imaging on a bounded executor** (2026-10-25). Original scope: (D3). `asyncio.to_thread` for decode and
      encode, pool sized to core count rather than anyio's default 40, and a semaphore
      capping concurrent decodes. Without this, D2 makes the Celeron slower.
- [ ] **W1.6 — One-shot data migration.** SQLite → Postgres on the Mac: 33,712 sources,
      70,002 findings with their embeddings, 77,977 scans, 36,884 assignments. Verify
      counts and spot-check the embedding round-trip.

      Non-destructive: it reads the SQLite file and writes a new database, so it can be
      re-run as often as needed. The resulting Postgres is what W2 dumps and ships to
      paiolo, which makes this the step that produces the deployable artifact.
- [x] **W1.7 — Concurrency machinery deleted** (2026-10-25, unavoidably, as part of
      W1.3). `connection.py` went **682 → 202 lines**: the shared connection, the
      `RLock`, `@_locked` on all 105 methods, `_TransactionalConnection` and its
      `_skip_commit` flag, and ~400 lines of on-open SQLite migrations —
      `_create_tables`, `_backfill_finding_scan_ids`, `_migrate_to_finding_assignment`,
      and `_enforce_finding_scan_id_not_null` with its `PRAGMA foreign_keys=OFF` rebuild.
      Alembic owns the schema (D4) and the data has already passed through those
      migrations, so retaining them would have been dead code guarding a database shape
      that no longer exists. Original item: The shared connection, the `RLock`, the
      `_locked` decorator, `_TransactionalConnection`. The `foreign_keys=OFF` rebuild
      hazard disappears with them.
- [x] **W1.8 — Test suite green on Postgres** (2026-10-25). **425 passed**, e2e included;
      ruff, ruff-format and mypy pass across all files. Five commits on
      `w1_postgres_async`, nothing pushed.

      > **Running the suite found four production bugs that mypy and review both missed**,
      > and one of them had the cluster detail page broken outright. **Five async
      > functions were called without `await`** — `rank_subjects_for_cluster`,
      > `compute_singleton_hints`, `compare_subjects`, `find_similar_unclustered`,
      > `compute_cluster_hint`. The coroutine went straight into a Jinja context, so there
      > was no unused value for mypy to flag, and two callers were still `def`. Swept the
      > package with an AST pass afterwards; those five were all of them. Also `MAX(id)`
      > on a uuid (no such aggregate), `export_json` unable to serialise UUID/datetime,
      > and `scans list` applying `:>6` to a uuid and `:24s` to a datetime.

      > **Test infrastructure, two findings worth keeping.** Rollback isolation cannot
      > work for tests that drive the app or the CLI: a second pool writing to tables the
      > test's open transaction holds deadlocks, and the suite hung until it was split
      > into `PostgresTestCase` (rollback) and `PostgresAppTestCase` (commit + truncate).
      > Connections now set `lock_timeout` and `statement_timeout` so that shape fails
      > fast rather than hanging. And **Playwright tests must sort last** — the sync API
      > holds a loop in the main thread, after which `IsolatedAsyncioTestCase` cannot
      > start its own, which had failed 43 downstream tests.

      **Behaviour that genuinely changed**, encoded in tests as decisions rather than
      ports: concurrent cluster dismissal can now yield **two** receipts, not one — the
      old "exactly one" came from the process-wide `RLock`, not from correctness, so the
      assertion is now the no-half-state invariant and the reentrancy test died with the
      lock. Orphan tests swapped `PRAGMA foreign_keys=OFF` for
      `session_replication_role = replica`, keeping the detector covered. `cluster_id` can
      no longer equal `subject_id` (uuid versus int32), so that collision test keeps its
      assertion and loses its premise. A malformed id in a URL is now **422** from path
      validation rather than 404 from the handler; those tests use a well-formed absent
      uuid so they still cover route ordering and missing resources.

      Original scope: with a Postgres fixture. 18 files, 5,692
      lines, 14 of them touching `FaceDB`.
- [ ] **W1.9 — Cutover, local only.** Stop `serve` on the Mac, run the migration with
      `--apply` against the local cluster, point the Mac at Postgres, keep the final
      SQLite snapshot as the rollback. **Rewritten 2026-10-25** — the original said "point
      *both machines* at Postgres", which assumed a network-reachable server. There is one
      machine in scope. Needs the user's explicit go-ahead.

## W1 follow-ups (found by `ritrova-3b`, 2026-10-25)

- [ ] **`sync_scan_types` has no caller in the application.** Defined at
      `connection.py:165`; `tests/conftest.py:52` is the only invocation. It used to run
      on every database open, so **the `scan_types` catalog is empty in any real
      deployment**. Nothing breaks today only because `scans.scan_type` is `text` with no
      `REFERENCES` — verified, precisely because an unseeded catalog behind a real foreign
      key would have been an outage waiting for the first scan. Either call it from a
      migration or delete it; leaving a seeding function that nothing seeds with is the
      worst of the three.
- [ ] **`imaging.py:1` cites "ADR-012 D3".** The D-numbers exist only in this plan, and
      this plan says of itself that it will be archived rather than maintained. The
      citation becomes unresolvable the moment that happens. Same class of defect as the
      ADR-006 gap, in a docstring — and the same reason CLAUDE.md forbids citing ticket
      keys in comments. Replace with the reason, or with a reference to something durable.

## W2 — Dockerise

- [ ] **W2.1 — `Dockerfile`.** `python:3.14-slim`, base install only, `uv sync --no-dev
      --frozen`, explicit `WORKDIR`, non-root user whose UID matches the owner of
      `/srv/family-data/Nextcloud`.
- [ ] **W2.2 — `compose.yaml`.** Postgres + app. Photos bind-mounted **read-only** (design
      principle #5), Postgres data on its own volume, image caches on a third, `restart:
      unless-stopped`.
- [ ] **W2.3 — Still one app process.** `UndoStore` keeps its single slot in process
      memory, so N workers means undo 404s for (N−1)/N of attempts — and every write is
      meant to be undoable (FEAT-5). Async makes one process sufficient; the point is that
      it stays load-bearing, not a tuning choice.
- [ ] **W2.4 — `/healthz`** plus a compose healthcheck.
- [ ] **W2.5 — Resource limits** so neither Ritrova nor Postgres can starve pihole. DNS for
      the whole house runs on this box.
- [ ] **W2.6 — Log driver with rotation.** Snap docker's defaults will fill the volume.
- [ ] **W2.7 — Firewall 8787 to the NAS.** Publish on `172.20.42.3:8787`, allow only
      `172.20.42.2`. Also decide what may reach Postgres: the Mac needs it, nothing else
      should.
- [ ] **W2.8 — Build on the Mac.** `buildx --platform linux/amd64`, then GHCR or
      `docker save | ssh … docker load`. A J4125 building wheels is a wasted evening.
- [ ] **W2.9 — `make tailwind`** to fetch the pinned standalone binary. The image doesn't
      need it (`style.css` is committed); a fresh clone currently can't build CSS at all.

## W3 — Server runmode

- [ ] **W3.1 — A real serve-only profile.** Manual-finding creation imports the detector
      lazily inside the request handler; on a base install that surfaces as a 500 from a
      raw `ModuleNotFoundError`. Make it a deliberate 503 with a clear message.
- [ ] **W3.2 — Remove the affordance, not just the endpoint.** Hide Shift+drag in the
      templates when the profile is server.
- [ ] **W3.3 — `analyse` on a base install should print an install hint,** not a raw
      `ModuleNotFoundError`.
- [ ] **W3.4 — Audit every other route for a model import.** `findings_service` is the
      known one; confirm there is no second crack.

## W4 — Performance on a J4125

A 2 GHz Celeron decoding 24-megapixel JPEGs is the difference between a service the family
uses and one they give up on. The database half of this is solved by W1; the imaging half
is not.

- [ ] **W4.1 — Cache resized source images.** Face thumbnails are cached to disk, but
      `/api/sources/{id}/image?max_size=1600` fully decodes the original on every request.
      Cache by (source_id, max_size).
- [ ] **W4.2 — `Image.draft()` for JPEG decode.** Pillow can decode directly at 1/2, 1/4 or
      1/8 scale instead of decoding full-size then resampling. Cheap, large, helps both
      paths.
- [ ] **W4.3 — Fix the per-request backfill on `/api/browse-html`.** It runs a full-table
      path-metadata backfill on every call, including every infinite-scroll continuation,
      against ~107 K sources. Move it to startup or to `backfill-path-metadata`.
- [ ] **W4.4 — Measure.** Cold-cache grid render and a `cluster` run over 70 K findings on
      paiolo, after W1. Numbers before further tuning.
- [ ] **W4.5 — Pre-warm the caches** overnight, if W4.4 says it's needed.

## W5 — Authentication

Spec is `docs/auth-session-design-home-server.md`. Nothing in it is built: no login, no
session, no CSRF, no user table anywhere in `src/`.

- [ ] **W5.1 — Tables** via Alembic: `users`, `allowed_users`, `sessions`, `audit_log`.
- [ ] **W5.2 — Google OIDC authorization-code flow.** Validate issuer, audience, expiry,
      `email_verified`. Key identity on `sub`, not email.
- [ ] **W5.3 — Server-side sessions.** Opaque token in the cookie, hash in the table.
      `HttpOnly` + `Secure` + `SameSite=Lax`. 12 h idle, 7 d absolute.
- [ ] **W5.4 — CSRF on every mutating request.** The largest single piece of work here:
      every `POST`/`PUT`/`PATCH`/`DELETE` route plus every `fetch()` in `static/js`.
- [ ] **W5.5 — Roles.** `admin` / `editor` / `viewer`.
- [ ] **W5.6 — UI:** login, unauthorized, forbidden, user menu, admin allowlist page.
- [ ] **W5.7 — In-app rate limiting** on login and callback. DSM's proxy has none, and the
      design doc assumed the proxy would provide it.
- [ ] **W5.8 — Audit log** for login success/failure, logout, role change, account disable,
      sensitive mutations.
- [ ] **W5.9 — Secrets.** Client id, client secret, session signing key and the Postgres
      password in a `0600` env file on the box. Never in the repo.
- [ ] **W5.10 — Publish the OAuth consent screen** once the scope list is confirmed to be
      only `openid`/`email`/`profile`.

## W6 — HTTPS and DNS

Blocked on W5. Each step is small; the ordering matters more than the effort.

- [ ] **W6.1 — Add `photos.papalini.casa` as a SAN** on the DSM Let's Encrypt certificate.
- [ ] **W6.2 — DSM Application Portal rule:** `photos.papalini.casa:443` →
      `http://172.20.42.3:8787`. The two existing rules both point at `localhost`; this is
      the first with a remote backend.
- [ ] **W6.3 — Public DNS A record** → 2.39.73.4.
- [ ] **W6.4 — Pihole split-horizon record** `photos.papalini.casa → 172.20.42.2` — the
      NAS, not paiolo, because TLS terminates there. Mirrors how `cloud` already resolves
      internally.
- [ ] **W6.5 — Proxy headers.** uvicorn with `--proxy-headers
      --forwarded-allow-ips=172.20.42.2`, plus `TrustedHostMiddleware`. Without it the OIDC
      redirect URI and the `Secure` cookie see the wrong scheme and host, and login breaks
      silently.
- [ ] **W6.6 — Raise the DSM proxy timeout** for large transfers. The print-selection zip
      export streams originals and can run to hundreds of megabytes.

## W7 — Operability

- [ ] **W7.1 — pytest markers.** There are none, so `pytest tests/` always pulls in the
      Playwright layer, which errors rather than skips without `playwright install
      chromium`. Needed before CI exists.
- [ ] **W7.2 — CI.** GitHub Actions: pre-commit plus the non-browser tests. There is none
      today.
- [ ] **W7.3 — Reconcile `make lint` with the pre-commit hook.** `make lint` runs ruff and
      mypy over `src/` and `tests/`; the hook runs ruff `--fix`, ruff-format and mypy over
      `src/` only. CLAUDE.md says checkers go through the hook, so `make lint` should call
      it.
- [ ] **W7.4 — `make deploy`.** Tag, build, ship, `alembic upgrade head`, restart, verify
      `/healthz`.
- [ ] **W7.5 — Nightly `pg_dump`** (D5), gzipped into the Nextcloud tree so the NAS carries
      it off-box. 14 daily + 8 weekly.
- [ ] **W7.6 — Restore drill.** Restore a dump into a scratch database and run `doctor` and
      `stats` against it. A backup nobody has restored is not a backup.
- [ ] **W7.7 — Logging for a container.** No structlog, and the CLI prints. Route uvicorn
      and app logs to stdout consistently; drop `print` from anything the server runs.
- [ ] **W7.8 — Rewrite the deployment half of the README.** Materially stale: flags that do
      not exist, a four-table schema, an `adr/` directory that is gone, no mention of
      `warmup`, `doctor` or `prune`.

## Scale (measured 2026-10-24)

**33,712 sources, 70,002 findings, 77,977 scans, 154 subjects, 36,884 assignments**
(35,382 to a subject, 1,502 exclusions). The Nextcloud tree holds roughly 107 K files
under `Photos/`, so a large part of the archive has never been analysed.

The assignments are the irreplaceable data — everything else can be recomputed from the
photos. They are also still accumulating on the Mac.

## Open questions

1. **Scope of "no curation on the server".** Taken to mean manual-finding creation
   (Shift+drag), which needs the detection stack. Naming, assigning, merging, dismissing
   and print selection need no models and stay enabled.
2. **The Windows machine.** `.env_windows` and the `*_data_win` directory name suggest a
   third analysing machine. Still in use? Under D1 it would just connect to Postgres too.
3. **The unanalysed remainder.** 33,712 sources against ~107 K files. Non-photo files and
   excluded directories, or genuinely unscanned years?
4. **Primary keys for the new auth tables.** CLAUDE.md requires UUIDv7 where UUIDs are
   generated. Existing tables use integer identities and are not worth churning; the auth
   tables are new and could go either way.

## Log

**2026-10-24 — the database write is explained, and nothing was lost.** Between 16:20 and
17:28 the Mac's `faces.db` mtime moved from `18 ott 18:46` to `24 ott 17:28` and the `-wal`
(833 KB) and `-shm` beside it disappeared.

Cause: a Ritrova process on the Mac was shut down cleanly. The evidence is inside the
database — `max(curated_at)` is `2026-10-24T14:46:00Z`, after the 16:20 observation and
before the 17:28 change. Something had the DB open read-write that afternoon, and SQLite
checkpoints the WAL and removes both sidecars on a clean close. No process holds the file
now. It was not a sync overwrite: a downloaded replacement would not contain that
afternoon's curation.

Verified against the Oct 18 run report (scan ids 134786–135030): all 245 scan rows and all
1,003 findings present. Cluster membership differs from the report — 706 against 732 —
which is expected, since `cluster` clears and recomputes per species and six days of
curation have happened since.

Two earlier readings of this event were wrong: it was neither an unidentified writer nor
the sync client dropping the WAL sidecars. The `sqlite3 -readonly` lead was also wrong —
the flag is supported; the open failed because a read-only open of a WAL database needs to
create `-shm` and the sandbox denies that write. Use `immutable=1` to read these files.

Separately and still open: paiolo's reconciliation resurrected the stale `ricorda_data` /
`ricorda_data_win` directories and propagated them to the Mac at 17:43 and 18:01. The sync
is stopped pending W0.3.

## Sources consulted

- `core-knowledge/02-architecture/` and `07-operations/` (Kipi, branch
  `kipi_core_knowledge_init`)
- `KIPI_FINDINGS.md` — per-process undo store, CWD-relative YOLO weights, the
  base-install cracks, the missing Tailwind binary, the unmarked Playwright tests, the
  `foreign_keys` rebuild hazard, the per-request browse backfill
- `docs/auth-session-design-home-server.md`
- Direct read-only inspection of 172.20.42.3 and 172.20.42.2, and of the live database,
  2026-10-24
