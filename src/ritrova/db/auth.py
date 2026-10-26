"""The sign-in allowlist (W5).

Only an address on this list may sign in. The list is deliberately separate
from ``users``: an entry is an invitation, and the ``users`` row appears on
first successful login, keyed on the Google ``sub`` rather than the address.

Addresses are stored lowercased. The list is typed by hand while Google's
OIDC claim always arrives lowercased, so comparing raw strings would let a
capitalised invitation silently fail to match the account it was written for —
denying a login that should succeed. ``users_email_lower_key`` and
``allowed_users_email_lower_key`` enforce the same rule in the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor

ROLES = ("admin", "editor", "viewer")

_ALLOWED_SELECT = """
    SELECT id, email, role, created_at, created_by
    FROM allowed_users
"""


@dataclass(frozen=True)
class AllowedUser:
    id: UUID
    email: str
    role: str
    created_at: datetime
    created_by: UUID | None


def normalise_email(email: str) -> str:
    return email.strip().lower()


def _to_allowed_user(row: object) -> AllowedUser:
    return AllowedUser(
        id=row[0],  # type: ignore[index]
        email=row[1],  # type: ignore[index]
        role=row[2],  # type: ignore[index]
        created_at=row[3],  # type: ignore[index]
        created_by=row[4],  # type: ignore[index]
    )


class AuthMixin(_DBAccessor):
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
