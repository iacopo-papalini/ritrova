"""Users, the sign-in allowlist, sessions and the audit log (W5).

Only an address on the allowlist may sign in. The list is deliberately
separate from ``users``: an entry is an invitation, and the ``users`` row
appears on first successful login, keyed on the Google ``sub`` rather than
the address.

Addresses are stored lowercased. The list is typed by hand while Google's
OIDC claim always arrives lowercased, so comparing raw strings would let a
capitalised invitation silently fail to match the account it was written
for — denying a login that should succeed. ``users_email_lower_key`` and
``allowed_users_email_lower_key`` enforce the same rule in the schema.

Sessions are stored as a digest of the cookie value, never the value
itself, so a reader of the table cannot mint a cookie from it. A session
dies on whichever comes first: explicit revocation, the absolute deadline
in ``expires_at``, or ``last_seen_at`` falling further behind than
``IDLE_TIMEOUT``.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor

ROLES = ("admin", "editor", "viewer")

IDLE_TIMEOUT = timedelta(hours=12)
ABSOLUTE_LIFETIME = timedelta(days=7)

# 32 bytes from the system CSPRNG: the cookie value is the whole credential,
# so it has to be unguessable rather than merely unique.
_TOKEN_BYTES = 32

_ALLOWED_SELECT = """
    SELECT id, email, role, created_at, created_by
    FROM allowed_users
"""

_USER_SELECT = """
    SELECT id, google_sub, email, display_name, role, is_active,
           created_at, last_login_at
    FROM users
"""


@dataclass(frozen=True)
class AllowedUser:
    id: UUID
    email: str
    role: str
    created_at: datetime
    created_by: UUID | None


@dataclass(frozen=True)
class User:
    id: UUID
    google_sub: str
    email: str
    display_name: str | None
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def may_write(self) -> bool:
        return self.role in ("admin", "editor")


def normalise_email(email: str) -> str:
    return email.strip().lower()


def new_session_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_session_token(token: str) -> bytes:
    """Digest the cookie value for storage and lookup.

    A plain SHA-256 rather than a password hash on purpose: the input is 32
    bytes of CSPRNG output, so there is no guessing to slow down, and the
    digest has to be cheap enough to compute on every request.
    """
    return hashlib.sha256(token.encode()).digest()


def _to_allowed_user(row: object) -> AllowedUser:
    return AllowedUser(
        id=row[0],  # type: ignore[index]
        email=row[1],  # type: ignore[index]
        role=row[2],  # type: ignore[index]
        created_at=row[3],  # type: ignore[index]
        created_by=row[4],  # type: ignore[index]
    )


def _to_user(row: object) -> User:
    return User(
        id=row[0],  # type: ignore[index]
        google_sub=row[1],  # type: ignore[index]
        email=row[2],  # type: ignore[index]
        display_name=row[3],  # type: ignore[index]
        role=row[4],  # type: ignore[index]
        is_active=row[5],  # type: ignore[index]
        created_at=row[6],  # type: ignore[index]
        last_login_at=row[7],  # type: ignore[index]
    )


class AuthMixin(_DBAccessor):
    # ── allowlist ────────────────────────────────────────────────────────

    async def allow_user(
        self,
        email: str,
        role: str = "editor",
        created_by: UUID | None = None,
    ) -> AllowedUser:
        """Invite ``email`` to sign in, or update the role of an existing entry."""
        address = normalise_email(email)
        if not address:
            msg = "Email cannot be empty."
            raise ValueError(msg)
        if role not in ROLES:
            msg = f"Unknown role {role!r}; expected one of {', '.join(ROLES)}."
            raise ValueError(msg)
        await self.execute(
            "INSERT INTO allowed_users (id, email, role, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (lower(email)) DO UPDATE SET role = EXCLUDED.role",
            uuid7(),
            address,
            role,
            self._now(),
            created_by,
        )
        invited = await self.find_allowed_user(address)
        if invited is None:  # pragma: no cover - the insert above guarantees a row
            msg = f"allowlist entry for {address} vanished immediately after write"
            raise RuntimeError(msg)
        return invited

    async def find_allowed_user(self, email: str) -> AllowedUser | None:
        row = await self.fetchrow(
            f"{_ALLOWED_SELECT} WHERE lower(email) = ?", normalise_email(email)
        )
        return _to_allowed_user(row) if row else None

    async def list_allowed_users(self) -> list[AllowedUser]:
        rows = await self.fetch(f"{_ALLOWED_SELECT} ORDER BY lower(email)")
        return [_to_allowed_user(r) for r in rows]

    async def revoke_allowed_user(self, email: str) -> bool:
        """Remove an invitation. Returns False if there was nothing to remove.

        Existing sessions are untouched: revoking an invitation stops future
        logins, and ending current ones is a separate, explicit action.
        """
        removed = await self.fetchval(
            "DELETE FROM allowed_users WHERE lower(email) = ? RETURNING id",
            normalise_email(email),
        )
        return removed is not None

    # ── users ────────────────────────────────────────────────────────────

    async def find_user_by_google_sub(self, google_sub: str) -> User | None:
        row = await self.fetchrow(f"{_USER_SELECT} WHERE google_sub = ?", google_sub)
        return _to_user(row) if row else None

    async def find_user_by_email(self, email: str) -> User | None:
        row = await self.fetchrow(f"{_USER_SELECT} WHERE lower(email) = ?", normalise_email(email))
        return _to_user(row) if row else None

    async def record_login(
        self,
        google_sub: str,
        email: str,
        display_name: str | None,
        role: str,
    ) -> User:
        """Create or refresh the local user behind a Google identity.

        Keyed on ``google_sub``: a Google account's email can change, its
        subject cannot. The address and display name are refreshed from the
        claims on every login so a changed email follows the identity rather
        than orphaning it.
        """
        address = normalise_email(email)
        now = self._now()
        await self.execute(
            "INSERT INTO users (id, google_sub, email, display_name, role, created_at, "
            "last_login_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (google_sub) DO UPDATE SET email = EXCLUDED.email, "
            "display_name = EXCLUDED.display_name, last_login_at = EXCLUDED.last_login_at",
            uuid7(),
            google_sub,
            address,
            display_name,
            role,
            now,
            now,
        )
        user = await self.find_user_by_google_sub(google_sub)
        if user is None:  # pragma: no cover - the insert above guarantees a row
            msg = f"user row for {google_sub} vanished immediately after write"
            raise RuntimeError(msg)
        return user

    async def set_user_role(self, user_id: UUID, role: str) -> None:
        if role not in ROLES:
            msg = f"Unknown role {role!r}; expected one of {', '.join(ROLES)}."
            raise ValueError(msg)
        await self.execute("UPDATE users SET role = ? WHERE id = ?", role, user_id)

    async def set_user_active(self, user_id: UUID, *, is_active: bool) -> None:
        """Enable or disable an account.

        Disabling also ends every session the user holds: an account that can
        still act until its cookies expire is not disabled.
        """
        await self.execute("UPDATE users SET is_active = ? WHERE id = ?", is_active, user_id)
        if not is_active:
            await self.revoke_user_sessions(user_id)

    async def list_users(self) -> list[User]:
        rows = await self.fetch(f"{_USER_SELECT} ORDER BY lower(email)")
        return [_to_user(r) for r in rows]

    # ── sessions ─────────────────────────────────────────────────────────

    async def create_session(
        self,
        user_id: UUID,
        token: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> UUID:
        session_id = uuid7()
        now = self._now()
        await self.execute(
            "INSERT INTO sessions (id, user_id, session_token_hash, created_at, "
            "last_seen_at, expires_at, user_agent, ip_address) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?::inet)",
            session_id,
            user_id,
            hash_session_token(token),
            now,
            now,
            now + ABSOLUTE_LIFETIME,
            user_agent,
            ip_address,
        )
        return session_id

    async def user_for_session_token(self, token: str) -> tuple[User, UUID] | None:
        """Resolve a cookie value to its user, or None if it is not usable.

        One query covers all four ways a session stops counting — revoked,
        past its absolute deadline, idle too long, or belonging to a disabled
        account — so none of them can be forgotten at a call site.
        """
        row = await self.fetchrow(
            "SELECT u.id, u.google_sub, u.email, u.display_name, u.role, u.is_active, "
            "       u.created_at, u.last_login_at, s.id "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.session_token_hash = ? "
            "  AND s.revoked_at IS NULL "
            "  AND s.expires_at > ? "
            "  AND s.last_seen_at > ? "
            "  AND u.is_active",
            hash_session_token(token),
            self._now(),
            self._now() - IDLE_TIMEOUT,
        )
        if row is None:
            return None
        return _to_user(row), row[8]

    async def touch_session(self, session_id: UUID) -> None:
        await self.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE id = ?", self._now(), session_id
        )

    async def revoke_session(self, session_id: UUID) -> None:
        await self.execute(
            "UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            self._now(),
            session_id,
        )

    async def revoke_user_sessions(self, user_id: UUID) -> int:
        revoked = await self.fetch(
            "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL "
            "RETURNING id",
            self._now(),
            user_id,
        )
        return len(revoked)

    async def delete_expired_sessions(self) -> int:
        """Drop rows no session could use again. Safe to run at any time."""
        gone = await self.fetch(
            "DELETE FROM sessions WHERE expires_at < ? RETURNING id", self._now()
        )
        return len(gone)

    # ── audit log ────────────────────────────────────────────────────────

    async def record_audit(
        self,
        action_type: str,
        *,
        actor_user_id: UUID | None = None,
        actor_address: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> UUID:
        """Append an audit row.

        ``actor_user_id`` is optional because the events most worth recording
        include ones with no user to attribute: a refused sign-in has an
        address and an email but no account. ``actor_address`` is what keeps
        such a row meaningful.
        """
        audit_id = uuid7()
        await self.execute(
            "INSERT INTO audit_log (id, actor_user_id, actor_address, action_type, "
            "entity_type, entity_id, payload, created_at) "
            "VALUES (?, ?, ?::inet, ?, ?, ?, ?::jsonb, ?)",
            audit_id,
            actor_user_id,
            actor_address,
            action_type,
            entity_type,
            entity_id,
            json.dumps(payload) if payload is not None else None,
            self._now(),
        )
        return audit_id

    async def recent_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.fetch(
            "SELECT a.id, a.actor_user_id, u.email AS actor_email, a.actor_address, "
            "       a.action_type, a.entity_type, a.entity_id, a.payload, a.created_at "
            "FROM audit_log a LEFT JOIN users u ON u.id = a.actor_user_id "
            "ORDER BY a.created_at DESC LIMIT ?",
            limit,
        )
        # asyncpg hands back jsonb as text. This function serialises on the
        # way in, so it parses on the way out rather than leaving every
        # caller to discover that a dict came back as a string.
        return [
            {**dict(r), "payload": json.loads(r["payload"]) if r["payload"] else None} for r in rows
        ]
