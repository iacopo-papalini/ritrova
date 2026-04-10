# Ritrova Auth And Session Design For Home-Server Deployment

Date: 2026-04-10

Deployment assumption:

- Ritrova runs on a small server in your home network
- exposed to the internet via your own domain
- TLS terminated with Let's Encrypt at the reverse proxy
- family-only access
- Google login in testing mode
- ML runs offline on a MacBook and results are synced/imported separately

## Recommendation Summary

Use this model:

- Google OIDC for login
- local user record in Ritrova
- allowlist of family email addresses
- secure cookie-based app sessions
- server-side CSRF protection for writes
- explicit logout
- idle timeout + absolute session lifetime
- audit log for important mutations

Do not use self-signed JWTs as the primary session mechanism unless you later split Ritrova into multiple services.

## Why Cookie Sessions Instead Of JWTs

For this deployment, cookie sessions are simpler and safer operationally:

- easier logout semantics
- easier revocation
- easier forced re-login
- less token-handling code
- no need for stateless distributed auth

JWTs solve problems you do not currently have:

- multiple backend services
- stateless horizontal scaling
- service-to-service auth

For a single-host family app, normal secure sessions are the right tool.

## Target Architecture

1. Browser hits `https://ritrova.yourdomain.tld`
2. Reverse proxy terminates TLS and forwards to Ritrova
3. User clicks `Sign in with Google`
4. Ritrova redirects to Google OIDC
5. Google returns auth code to Ritrova callback
6. Ritrova exchanges code for tokens
7. Ritrova validates the Google ID token
8. Ritrova checks allowlist / user record
9. Ritrova creates local session
10. Ritrova sets secure session cookie
11. Subsequent requests use Ritrova session, not Google token directly

## Authentication Flow

Use OpenID Connect, not plain OAuth profile-fetching.

At login:

- use Google OIDC authorization code flow
- validate:
  - issuer
  - audience
  - expiration
  - subject (`sub`)
  - email verified
- map Google identity to a local user

Persist locally:

- `google_sub`
- `email`
- `display_name`
- `role`
- `is_active`
- timestamps

Important:

- identity should be keyed by `google_sub`, not just email
- email is useful for allowlisting and display, but `sub` is the stable identity key

## Authorization Model

Keep it minimal.

Suggested roles:

- `admin`
  - manage users
  - change settings
  - launch imports/sync jobs
  - all editor actions
- `editor`
  - assign, unassign, merge, dismiss, rename
  - browse all content
- `viewer`
  - read-only access

For family use, you may start with everyone as `editor` except one `admin`.

## Allowlist Model

Use an explicit allowlist.

Recommended rule:

- only users whose Google account email is in the local allowlist may sign in

Store:

- `email`
- optional `google_sub` once first login succeeds
- role
- invited_by
- invited_at

This is better than “any Google account can log in”.

## Session Design

Use secure cookie-backed sessions.

Cookie properties:

- `HttpOnly`
- `Secure`
- `SameSite=Lax`
- scoped to your app domain

Session contents should be minimal:

- session ID
- user ID
- issued-at
- last-seen
- CSRF secret or reference

Recommended storage models:

### Option A: Server-side session table

Best default.

Cookie stores:

- opaque random session ID

Server stores:

- session ID hash
- user ID
- created_at
- last_seen_at
- expires_at
- IP / user agent metadata if desired
- revoked_at

Pros:

- easy logout
- easy revoke-all-sessions
- easy forced reauth
- simple mental model

### Option B: Signed cookie session

Also acceptable for a small app if you want less DB/session plumbing.

Cookie stores signed session payload.

Pros:

- simpler infra

Cons:

- session invalidation is weaker unless you add versioning or denylist logic

Recommendation:

- use Option A if you are already adding users/audit tables
- only use signed cookie sessions if you want minimal implementation overhead

## Session Lifetime

Use both idle and absolute expiration.

Suggested defaults:

- idle timeout: 12 hours
- absolute lifetime: 7 days

That aligns well with the Google testing-mode re-consent window anyway.

Behavior:

- if idle timeout exceeded, require login again
- if absolute lifetime exceeded, require login again
- optionally extend `last_seen_at` on activity

## Logout

Logout should:

- delete/revoke the local session
- clear the session cookie

You do not need to log the user out of Google itself.

Also add:

- `logout all sessions`
- admin revoke for a user if needed

## CSRF Protection

Required for all state-changing browser actions.

Since Ritrova is a cookie-authenticated app, CSRF protection matters for:

- POST
- PUT
- PATCH
- DELETE

Recommended pattern:

- per-session CSRF token
- inject token into templates
- submit via hidden input or header
- validate on every mutating request

This applies to:

- assign
- unassign
- merge
- dismiss
- rename
- delete
- future task triggers

## Rate Limiting

Add rate limiting especially for:

- login start endpoint
- auth callback endpoint
- any future password fallback endpoints if added

For your use case, lightweight rate limiting is enough:

- per-IP
- per-email/login target where applicable

You can do this at:

- reverse proxy layer, or
- app middleware layer

Reverse proxy rate limiting is usually simpler and more robust.

## Reverse Proxy Responsibilities

Your reverse proxy should handle:

- TLS termination
- HTTP to HTTPS redirect
- HSTS if you are comfortable enforcing HTTPS
- request size limits
- basic rate limiting
- forwarded headers to the app

The app should still enforce:

- trusted hosts
- secure cookies
- session validation

## App Security Controls Relevant To Auth

Inside Ritrova, add:

- trusted host validation
- secure session cookie config
- no-cache headers on auth responses where appropriate
- audit log for:
  - login success
  - login failure
  - logout
  - role changes
  - account disable
  - sensitive mutations

## Suggested Database Tables

Minimum new tables:

### `users`

- `id`
- `google_sub`
- `email`
- `display_name`
- `role`
- `is_active`
- `created_at`
- `last_login_at`

### `allowed_users`

- `id`
- `email`
- `role`
- `created_at`
- `created_by`

You can collapse this into `users` if you prefer invitation-on-first-login semantics.

### `sessions`

- `id`
- `user_id`
- `session_token_hash`
- `created_at`
- `last_seen_at`
- `expires_at`
- `revoked_at`
- optional `user_agent`
- optional `ip_address`

### `audit_log`

- `id`
- `actor_user_id`
- `action_type`
- `entity_type`
- `entity_id`
- `payload_json`
- `created_at`

## UI Changes Needed

Add:

- login page / login button
- logout action
- unauthorized page
- forbidden page
- small user menu in nav
- admin page for:
  - allowlist management
  - user roles
  - active sessions

Optional but useful:

- “last signed in as ...”
- recent security activity in admin

## Recommended Libraries / Implementation Direction

At a high level, you need:

- OIDC client support
- session middleware or custom session handling
- CSRF middleware or custom token validation

Implementation approach:

- validate Google callback server-side
- create your own local session
- keep Google tokens out of normal app-page requests

Do not use Google access tokens as your app session.

## What To Avoid

- using only email string matching without verifying ID token claims
- storing raw session tokens in DB
- long-lived non-expiring sessions
- no logout path
- no CSRF because “only family uses it”
- exposing the app publicly without allowlist checks
- using self-signed JWTs for sessions just because they sound modern

## Practical Recommendation For Your Exact Case

Best fit:

- Google OIDC in testing mode
- explicit family email allowlist
- server-side session table
- secure `HttpOnly` cookie
- 12h idle timeout
- 7d absolute timeout
- explicit logout
- CSRF protection on all writes
- reverse-proxy rate limiting
- audit log for auth and data mutations

This gives you:

- no new password to remember
- no separate Ritrova MFA burden
- reasonable security posture for internet exposure
- simple operational model on one home server

## Future Upgrade Path

If your needs grow later:

- move from Google testing-mode to published app or a self-hosted IdP
- add account linking for non-Google users if needed
- add WebAuthn/passkeys for admins
- add VPN-only admin routes

But none of that is required for the initial deployment.

