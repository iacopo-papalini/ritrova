"""Tests for the sign-in allowlist (W5).

The case-insensitivity cases are the ones with teeth: a capitalised
invitation that fails to match Google's lowercased email claim would deny a
login that should succeed, and a second entry differing only in case would
make the allowlist ambiguous.
"""

from __future__ import annotations

from ritrova.db.auth import normalise_email

from ._postgres import PostgresTestCase

MINE = "iacopo.papalini@example.com"


class TestNormaliseEmail(PostgresTestCase):
    def test_strips_and_lowercases(self) -> None:
        self.assertEqual(normalise_email("  Iacopo.Papalini@Example.COM "), MINE)


class TestAllowlist(PostgresTestCase):
    async def test_allow_then_find(self) -> None:
        invited = await self.db.allow_user(MINE, role="admin")
        self.assertEqual(invited.email, MINE)
        self.assertEqual(invited.role, "admin")
        self.assertIsNone(invited.created_by)

        found = await self.db.find_allowed_user(MINE)
        assert found is not None
        self.assertEqual(found.id, invited.id)

    async def test_default_role_is_editor(self) -> None:
        invited = await self.db.allow_user(MINE)
        self.assertEqual(invited.role, "editor")

    async def test_lookup_is_case_insensitive(self) -> None:
        """Google's claim arrives lowercased; the invitation may not be."""
        await self.db.allow_user("Iacopo.Papalini@Example.COM", role="admin")
        found = await self.db.find_allowed_user(MINE)
        assert found is not None
        self.assertEqual(found.role, "admin")

    async def test_re_inviting_updates_the_role_instead_of_duplicating(self) -> None:
        first = await self.db.allow_user(MINE, role="viewer")
        second = await self.db.allow_user("IACOPO.PAPALINI@example.com", role="admin")
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.role, "admin")
        self.assertEqual(len(await self.db.list_allowed_users()), 1)

    async def test_unknown_role_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            await self.db.allow_user(MINE, role="superuser")

    async def test_empty_email_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            await self.db.allow_user("   ")

    async def test_stored_address_is_lowercased(self) -> None:
        await self.db.allow_user("Iacopo.Papalini@Example.COM")
        stored = await self.db.fetchval("SELECT email FROM allowed_users")
        self.assertEqual(stored, MINE)

    async def test_list_is_ordered_by_address(self) -> None:
        for address in ("zoe@example.com", "Adam@example.com", "mia@example.com"):
            await self.db.allow_user(address)
        listed = [u.email for u in await self.db.list_allowed_users()]
        self.assertEqual(listed, ["adam@example.com", "mia@example.com", "zoe@example.com"])

    async def test_revoke(self) -> None:
        await self.db.allow_user(MINE)
        self.assertTrue(await self.db.revoke_allowed_user("IACOPO.PAPALINI@EXAMPLE.COM"))
        self.assertIsNone(await self.db.find_allowed_user(MINE))

    async def test_revoking_a_missing_entry_reports_false(self) -> None:
        self.assertFalse(await self.db.revoke_allowed_user("nobody@example.com"))

    async def test_find_missing_returns_none(self) -> None:
        self.assertIsNone(await self.db.find_allowed_user("nobody@example.com"))
