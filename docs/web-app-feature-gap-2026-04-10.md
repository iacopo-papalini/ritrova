# Ritrova Web App Feature Gap

Date: 2026-04-10

Scope: gap analysis for evolving Ritrova from a strong local/single-user review tool into a real web application. This focuses on product/platform capability gaps such as authentication, collaboration, background jobs, deployment readiness, observability, and scalability. It is not a security review.

## Executive Summary

Ritrova is already a solid application in the product sense:

- it has a real domain model
- it has meaningful workflows
- it has a coherent web UI
- it has enough backend structure to support extension

But today it is still fundamentally a local operator tool:

- single process
- single SQLite database
- local filesystem as primary storage
- CLI-first ingestion and maintenance
- no user model
- no sessions or permissions
- no background task system
- no operational control plane

That is the main distinction.

To become a real web app, Ritrova needs six capabilities:

1. Identity and access management
2. Multi-user aware data model and action model
3. Background jobs and long-running workflow orchestration
4. Production-grade deployment and storage architecture
5. Observability and operational tooling
6. API and frontend boundaries that support growth

## Current State Summary

Based on the current codebase:

- backend: FastAPI app with server-rendered Jinja templates
- storage: SQLite with WAL and a process-local lock
- media: local filesystem paths and local thumbnail cache
- compute workflows: invoked from CLI, not from the web app
- UI updates: mostly htmx/Alpine plus direct fetch calls
- users: none
- auth: none
- sessions: none
- jobs: none
- tenant/workspace model: none
- realtime status: none
- metrics/admin: none

This is a strong local architecture, but not yet a web application architecture.

## Gap 1: Authentication And User Identity

### What is missing

- sign-in and sign-out
- user accounts
- session management
- persistent identity across requests
- user preferences
- ownership of actions

### Why it matters

Without authentication, the app cannot support:

- hosted usage
- shared family/team usage
- protected remote access
- personalization
- auditability of who changed what

### Minimum viable implementation

- email/password or passwordless login
- server-side sessions or signed cookies
- `users` table
- `created_by` / `updated_by` on important entities and actions
- per-user preferences:
  - default view
  - people/pets scope
  - review queue state

### Recommended direction

Start with single-tenant authentication, not multi-tenant SaaS complexity.

Pragmatic phase 1:

- one workspace
- multiple invited users
- simple roles:
  - owner
  - editor
  - viewer

This is enough to turn Ritrova into a real shared app without overdesigning it.

## Gap 2: Authorization, Roles, And Change Ownership

### What is missing

- no distinction between readers and editors
- no approval model for destructive operations
- no actor attribution on changes
- no action history per user

### Why it matters

As soon as more than one person can use the app, action ownership matters:

- who renamed a person
- who merged identities
- who dismissed detections
- who launched scans or cleanup

### Minimum viable implementation

- role-based permissions
- `audit_log` table for writes
- UI surfacing:
  - “last changed by”
  - “last reviewed by”
  - timestamps for edits

### Product note

This is not just admin plumbing. It changes confidence in collaborative curation.

## Gap 3: Workspace / Library Model

### What is missing

The app currently assumes one database and one photo root.

There is no first-class concept of:

- library
- workspace
- archive
- household/team collection

### Why it matters

A real web app eventually needs a unit of organization above the raw DB:

- one user may manage multiple libraries
- one family may share one archive
- different archives may need different scanning rules or naming conventions

### Minimum viable implementation

Introduce a `workspace` concept:

- workspace metadata
- storage root / source config
- members
- settings

Then scope core records conceptually to a workspace:

- persons
- photos
- faces
- tasks
- action logs

### Recommended direction

Even if you stay single-workspace at first, design the schema and service layer so that workspace scoping can be added cleanly.

## Gap 4: Background Jobs And Long-Running Task Orchestration

### What is missing

This is one of the biggest platform gaps.

Current heavy operations are CLI-driven:

- scan
- scan-videos
- scan-pets
- cluster
- auto-assign
- cleanup
- backfill-gps
- describe

The web app does not yet own those workflows.

### Why it matters

A real web app cannot depend on shell access for core product value.

Users need to be able to:

- launch jobs from the UI
- see status
- see progress
- cancel or retry
- inspect logs/errors
- understand what changed after completion

### Minimum viable implementation

This aligns with `FEAT-9` in `docs/bugs.md`.

Needed components:

- `jobs` table
- job states:
  - queued
  - running
  - completed
  - failed
  - canceled
- background worker process
- progress updates via polling or SSE
- job logs
- job result summary

### Recommended architecture

Phase 1:

- single in-app worker process
- one job at a time
- DB-backed job state
- polling or SSE

Phase 2:

- separate worker process
- task queue
- retry support
- concurrency controls by task type

This is the clearest boundary between “local tool” and “web app”.

## Gap 5: Storage Architecture

### What is missing

Current storage assumptions are local:

- images live on local disk
- thumbnails are cached to local disk
- DB stores paths
- video frames are generated locally

### Why it matters

This works well on one machine, but it becomes fragile for:

- hosted deployment
- containers
- scaling across processes or machines
- backups and migration

### Minimum viable implementation

Define storage abstractions for:

- source media
- generated derivatives
- temporary job artifacts

### Recommended direction

Phase 1:

- keep original photo library local or mounted
- formalize app-managed storage under a configured app data root
- centralize path resolution and derivative management

Phase 2:

- object storage for derivatives and generated assets
- optional remote source media support
- signed URL or proxy model for image delivery

### Important note

You do not need cloud object storage immediately. You do need a storage model that is not implicitly tied to one dev machine layout.

## Gap 6: Database Scalability And Data Model Maturity

### What is missing

SQLite is good for the current stage, but the app lacks features usually expected once usage grows:

- migration framework with explicit versioning
- operational admin tables
- action/event logs
- job tables
- session tables if server-side sessions are used
- richer indexing for search and retrieval

### Why it matters

The issue is not just query speed. It is operational scope.

As soon as you add:

- users
- jobs
- audit logs
- search indexes
- tags/descriptions
- undo state

the database becomes part of the product platform, not just the ML result store.

### Recommended direction

Short term:

- keep SQLite if deployment remains single-host
- add formal migrations
- add platform tables
- add health/admin queries

Medium term:

- plan a PostgreSQL migration path

Trigger points for moving off SQLite:

- multiple app instances
- concurrent background worker plus web server load
- hosted multi-user access
- richer search and filtering

SQLite is not the immediate problem. Lack of a migration path is.

## Gap 7: Search And Retrieval Infrastructure

### What is missing

Current search is narrow and mostly name-oriented.

Open roadmap items already point toward broader retrieval:

- `FEAT-6`: multi-person/pet photo search
- `FEAT-7`: generic metadata search
- `FEAT-8`: scene descriptions and tags

### Why it matters

A real web app needs a query model, not just a page with a search box.

Users should eventually be able to search across:

- people
- pets
- co-occurrence
- date ranges
- location
- file path / album
- generated tags
- generated descriptions
- reviewed vs unreviewed states

### Minimum viable implementation

- unified search schema
- filterable results model
- indexed metadata tables
- reusable query service layer

### Recommended direction

Treat search as a platform feature, not a template-specific form.

That means:

- clear query inputs
- reusable backend query objects/services
- result types:
  - persons
  - photos
  - tasks
  - maybe later events/edits

## Gap 8: Realtime Status And User Feedback Loop

### What is missing

- no live task progress
- no app-wide toast system
- no durable notifications
- no activity feed
- no “someone else changed this” awareness

### Why it matters

A web app needs a time dimension.

Users expect the system to tell them:

- what is running
- what just finished
- what failed
- what changed since they last looked

### Minimum viable implementation

- toasts for write operations
- global job status panel
- SSE or polling for background tasks
- recent activity list

### Recommended direction

Build a simple events layer:

- job events
- action events
- notification events

This unlocks both UX polish and future observability.

## Gap 9: API Design And Frontend Boundary

### What is missing

The app currently mixes:

- server-rendered pages
- HTML fragment endpoints
- JSON endpoints
- direct fetch calls in templates

That is acceptable now, but growth will get messy without clearer boundaries.

### Why it matters

A real web app benefits from a more intentional contract between frontend and backend:

- easier testing
- easier evolution
- clearer client behavior
- easier future UI changes

### Recommended direction

You do not need to rewrite into an SPA.

A pragmatic target is:

- server-rendered app remains primary
- JSON APIs become explicit service endpoints
- HTML fragment endpoints remain for htmx flows
- shared action conventions:
  - success payload shape
  - validation error shape
  - toast/event hooks

### Minimum viable implementation

- standard response contract for mutations
- standard error payload
- central action helpers in frontend JS
- consistent route model

## Gap 10: Observability And Operations

### What is missing

There is not yet a real operations layer:

- structured application logs
- request metrics
- task metrics
- job failure diagnostics
- health endpoints
- startup diagnostics
- admin/status dashboard

### Why it matters

Once the app runs outside a dev shell, operational visibility becomes essential.

You need to answer:

- Is the app healthy?
- Is the worker healthy?
- Which jobs fail most?
- How long do scans take?
- Why are thumbnails or photo loads slow?

### Minimum viable implementation

- structured logging
- request ID / job ID correlation
- `/health` and `/ready` endpoints
- basic metrics:
  - request latency
  - error count
  - job duration
  - queue depth
  - thumbnail cache hit rate

### Recommended direction

This can start very simply. The important thing is to add an operations surface before adding hosted complexity.

## Gap 11: Deployment And Environment Management

### What is missing

- no deployment profile separation
- no container/deployment story in the product surface
- no clear production config model beyond env vars
- no backup/restore workflow for app-managed data
- no admin controls for storage roots, model assets, or workers

### Why it matters

A real web app is not just code that can run on a server. It needs a supported operating model.

### Minimum viable implementation

- documented deployment topology
- environment profiles:
  - local
  - single-host production
  - hosted future
- backup strategy for:
  - DB
  - generated assets
  - logs or job history

### Recommended topology

Phase 1 production:

- one web process
- one worker process
- one shared app data volume
- SQLite or PostgreSQL depending on target scale

That is enough for a credible “real app” deployment.

## Gap 12: Collaboration Features

### What is missing

Even after auth, the app would still be single-operator in spirit unless it gains collaboration primitives:

- activity feed
- comments/notes
- review status
- assignment ownership
- conflict awareness

### Why it matters

For a family or team archive, collaboration is part of the value:

- one person scans
- another names
- another reviews pets
- another fixes wrong merges

### Minimum viable implementation

- reviewed/unreviewed markers
- last touched by
- notes on person/photo
- assignment comments

### Recommended direction

Do not jump to Google Docs-style real-time editing. Start with asynchronous collaboration.

## Gap 13: Admin And Data Governance Features

### What is missing

A real app needs some admin-level product functions:

- manage users
- manage roles
- inspect jobs
- inspect storage usage
- rebuild thumbnails
- re-run specific processing tasks
- import/export workflows
- backup/restore guidance

### Why it matters

Without an admin surface, the product remains dependent on direct shell/database intervention.

### Minimum viable implementation

- admin/settings page
- jobs page
- storage/settings diagnostics
- exports with metadata

## Gap 14: Performance And Scalability Strategy

### Current likely ceiling

The current architecture should work well for:

- one machine
- one active user
- moderate archive size
- background tasks run manually

### What will break first

Likely pressure points:

- SQLite write contention once jobs and web mutations overlap
- thumbnail generation and image serving under concurrent load
- local disk assumptions in containerized deployment
- long-running model tasks competing with web responsiveness
- in-process state assumptions once multiple workers/processes exist

### Scalability roadmap

Phase 1: single-host robustness

- keep one web node
- add one worker
- isolate heavy jobs from request serving
- formalize derivative storage
- add metrics

Phase 2: moderate multi-user hosting

- move to PostgreSQL
- move derivatives to shared/object storage
- externalize job execution
- add cache strategy for thumbnails and hot images

Phase 3: broader hosted product

- multi-workspace support
- horizontal scaling of stateless web nodes
- distributed workers
- formal search/indexing subsystem

## Gap 15: Productized Onboarding And Setup

### What is missing

Today setup is still operator-centric:

- environment variables
- CLI commands
- manual model downloads
- implicit app data conventions

### Why it matters

A real web app needs an onboarding story for non-developer users.

### Minimum viable implementation

- first-run setup flow
- library selection/config screen
- model readiness checks
- storage validation
- scan kickoff from UI

### Recommended direction

The ideal new-user flow should be:

1. Sign in
2. Create workspace
3. Connect/select photo library
4. Run initial scan from the UI
5. Watch progress
6. Start review

Right now the app begins at step 6.

## Highest-Priority Feature Gaps

If the goal is “make this a real web app” without boiling the ocean, these are the top priorities:

1. Authentication and roles
2. Web-triggered background jobs with progress
3. Workspace/library abstraction
4. Audit log plus undo/event system
5. Observability basics: health, logs, job metrics
6. Deployment/storage model for single-host production

Those six move Ritrova from “great local app” to “credible hosted/shared application”.

## Suggested Phased Plan

## Phase 1: Web App Foundation

- Add user accounts and sessions
- Add roles
- Add audit log
- Add settings/admin page
- Add formal migrations

## Phase 2: Operationalize Core Workflows

- Add jobs table and worker
- Add web-triggered scan/cluster/cleanup flows
- Add task progress UI
- Add notifications/toasts
- Add job history

## Phase 3: Production Hardening

- Formalize storage layout
- Add health/ready endpoints
- Add metrics and structured logs
- Add backup/export/restore workflows
- Decide SQLite vs PostgreSQL based on deployment target

## Phase 4: Collaboration And Retrieval

- Add multi-user review semantics
- Add richer search (`FEAT-6`, `FEAT-7`, `FEAT-8`)
- Add reviewed status, notes, and activity feed
- Add better admin tooling

## Bottom Line

Ritrova does not need a frontend rewrite to become a real web app.

It needs platform capabilities around the existing product:

- identity
- jobs
- storage model
- operational tooling
- collaboration primitives
- scalable persistence strategy

The current codebase is already good enough to support that evolution. The next step is not “more ML features first”. The next step is building the application shell around the product that already exists.

