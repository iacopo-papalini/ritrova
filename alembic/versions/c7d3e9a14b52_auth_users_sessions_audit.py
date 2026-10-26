"""users, allowed_users, sessions and audit_log

The four tables behind Google OIDC login (W5). Surrogate keys are UUIDv7 from
``uuid6.uuid7`` like every other table, and the aware audit columns are
``timestamptz`` per D9.

Decisions worth knowing, because the obvious alternative is wrong:

``users`` is keyed on ``google_sub``, not email. A Google account's email can
change; ``sub`` cannot. Email is still unique, but as a display and allowlist
key rather than an identity.

Email uniqueness is enforced on ``lower(email)``, not on the raw column. The
allowlist is typed by hand, and ``Iacopo@`` must not become a second account —
or, worse, miss an allowlist entry written in a different case and so be denied
a login that should succeed.

``sessions.session_token_hash`` stores a digest, never the token. A reader of
the table must not be able to mint a cookie from it. Lookup is by that hash, so
the UNIQUE constraint's own index serves it and no second index is added — the
mistake b2c41f08ad77 removed.

``sessions.expires_at`` is the absolute deadline only. The idle timeout is
``last_seen_at`` plus a policy interval, and policy belongs in configuration
where it can be changed without a migration.

``audit_log.actor_user_id`` is nullable and ``ON DELETE SET NULL``, both
deliberately. Nullable because a failed login has no user to attribute, and
``SET NULL`` rather than ``CASCADE`` because deleting a user must not erase the
record of what they did — a cascade here would make the audit log destructible
by the very action it exists to record. ``actor_address`` is kept alongside so a
row still says where an unattributable event came from.

``audit_log.entity_id`` is ``text`` rather than ``uuid`` because not every
entity has a UUID: a cluster id is an integer the clustering algorithm emits,
as the baseline notes for ``cluster_findings.cluster_id``. One column has to
hold both.

The spec calls the payload column ``payload_json``; it is ``payload`` here,
since ``jsonb`` already says what it holds.

Revision ID: c7d3e9a14b52
Revises: b2c41f08ad77
Create Date: 2026-10-26

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c7d3e9a14b52"
down_revision: str | Sequence[str] | None = "b2c41f08ad77"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE users (
            id            uuid PRIMARY KEY,
            google_sub    text NOT NULL UNIQUE,
            email         text NOT NULL,
            display_name  text,
            role          text NOT NULL DEFAULT 'editor',
            is_active     boolean NOT NULL DEFAULT true,
            created_at    timestamptz NOT NULL DEFAULT now(),
            last_login_at timestamptz,
            CHECK (role IN ('admin', 'editor', 'viewer'))
        )
    """)
    op.execute("CREATE UNIQUE INDEX users_email_lower_key ON users (lower(email))")

    # created_by is nullable: the first admin is seeded before any user row
    # exists, so it has no inviter to point at.
    op.execute("""
        CREATE TABLE allowed_users (
            id         uuid PRIMARY KEY,
            email      text NOT NULL,
            role       text NOT NULL DEFAULT 'editor',
            created_at timestamptz NOT NULL DEFAULT now(),
            created_by uuid REFERENCES users(id) ON DELETE SET NULL,
            CHECK (role IN ('admin', 'editor', 'viewer'))
        )
    """)
    op.execute("CREATE UNIQUE INDEX allowed_users_email_lower_key ON allowed_users (lower(email))")

    op.execute("""
        CREATE TABLE sessions (
            id                 uuid PRIMARY KEY,
            user_id            uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            session_token_hash bytea NOT NULL UNIQUE,
            created_at         timestamptz NOT NULL DEFAULT now(),
            last_seen_at       timestamptz NOT NULL DEFAULT now(),
            expires_at         timestamptz NOT NULL,
            revoked_at         timestamptz,
            user_agent         text,
            ip_address         inet
        )
    """)
    # "Log out everywhere" and the admin session list both read by user.
    op.execute("CREATE INDEX idx_sessions_user ON sessions(user_id)")

    op.execute("""
        CREATE TABLE audit_log (
            id            uuid PRIMARY KEY,
            actor_user_id uuid REFERENCES users(id) ON DELETE SET NULL,
            actor_address inet,
            action_type   text NOT NULL,
            entity_type   text,
            entity_id     text,
            payload       jsonb,
            created_at    timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP TABLE IF EXISTS sessions")
    op.execute("DROP TABLE IF EXISTS allowed_users")
    op.execute("DROP TABLE IF EXISTS users")
