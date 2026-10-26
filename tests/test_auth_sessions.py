"""Tests for users, sessions and the audit log (W5).

The session cases assert the four independent ways a session stops
counting. Each is a way an account could otherwise stay usable after it
should not: revoked, past its absolute deadline, idle too long, or
belonging to a disabled user.
"""

from __future__ import annotations

from datetime import timedelta

from ritrova.db.auth import (
    ABSOLUTE_LIFETIME,
    IDLE_TIMEOUT,
    hash_session_token,
    new_session_token,
)

from ._postgres import PostgresTestCase

SUB = "google-sub-12345"
MINE = "iacopo.papalini@example.com"


class TestSessionToken(PostgresTestCase):
    def test_tokens_are_unguessable_and_distinct(self) -> None:
        tokens = {new_session_token() for _ in range(100)}
        self.assertEqual(len(tokens), 100)
        self.assertGreaterEqual(min(len(t) for t in tokens), 32)

    def test_hash_is_stable_and_not_the_token(self) -> None:
        token = new_session_token()
        self.assertEqual(hash_session_token(token), hash_session_token(token))
        self.assertNotIn(token.encode(), hash_session_token(token))


class TestRecordLogin(PostgresTestCase):
    async def test_creates_the_user(self) -> None:
        user = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        self.assertEqual(user.google_sub, SUB)
        self.assertEqual(user.email, MINE)
        self.assertEqual(user.role, "admin")
        self.assertTrue(user.is_active)
        self.assertIsNotNone(user.last_login_at)

    async def test_second_login_reuses_the_row(self) -> None:
        first = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        second = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(await self.db.list_users()), 1)

    async def test_a_changed_google_email_follows_the_identity(self) -> None:
        """Keyed on sub, so a new address updates the row instead of orphaning it."""
        first = await self.db.record_login(SUB, MINE, "Iacopo", "editor")
        moved = await self.db.record_login(SUB, "new.address@example.com", "Iacopo P", "editor")
        self.assertEqual(first.id, moved.id)
        self.assertEqual(moved.email, "new.address@example.com")
        self.assertEqual(moved.display_name, "Iacopo P")
        self.assertEqual(len(await self.db.list_users()), 1)

    async def test_address_is_stored_lowercased(self) -> None:
        user = await self.db.record_login(SUB, "Iacopo.Papalini@Example.COM", None, "editor")
        self.assertEqual(user.email, MINE)

    async def test_role_helpers(self) -> None:
        admin = await self.db.record_login("s-admin", "a@example.com", None, "admin")
        editor = await self.db.record_login("s-editor", "e@example.com", None, "editor")
        viewer = await self.db.record_login("s-viewer", "v@example.com", None, "viewer")
        self.assertTrue(admin.is_admin)
        self.assertFalse(editor.is_admin)
        self.assertTrue(editor.may_write)
        self.assertFalse(viewer.may_write)


class TestSessions(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.user = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        self.token = new_session_token()
        self.session_id = await self.db.create_session(
            self.user.id, self.token, user_agent="pytest", ip_address="172.20.42.50"
        )

    async def test_a_fresh_session_resolves(self) -> None:
        found = await self.db.user_for_session_token(self.token)
        assert found is not None
        user, session_id = found
        self.assertEqual(user.id, self.user.id)
        self.assertEqual(session_id, self.session_id)

    async def test_the_raw_token_is_not_stored(self) -> None:
        stored = await self.db.fetchval("SELECT session_token_hash FROM sessions")
        self.assertEqual(stored, hash_session_token(self.token))
        self.assertNotEqual(stored, self.token.encode())

    async def test_an_unknown_token_resolves_to_nothing(self) -> None:
        self.assertIsNone(await self.db.user_for_session_token(new_session_token()))

    async def test_revoked_session_stops_working(self) -> None:
        await self.db.revoke_session(self.session_id)
        self.assertIsNone(await self.db.user_for_session_token(self.token))

    async def test_session_past_its_absolute_deadline_stops_working(self) -> None:
        await self.db.execute(
            "UPDATE sessions SET expires_at = ? WHERE id = ?",
            self.db._now() - timedelta(seconds=1),  # noqa: SLF001
            self.session_id,
        )
        self.assertIsNone(await self.db.user_for_session_token(self.token))

    async def test_idle_session_stops_working(self) -> None:
        await self.db.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
            self.db._now() - IDLE_TIMEOUT - timedelta(minutes=1),  # noqa: SLF001
            self.session_id,
        )
        self.assertIsNone(await self.db.user_for_session_token(self.token))

    async def test_touch_keeps_an_idle_session_alive(self) -> None:
        await self.db.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
            self.db._now() - IDLE_TIMEOUT + timedelta(minutes=5),  # noqa: SLF001
            self.session_id,
        )
        await self.db.touch_session(self.session_id)
        self.assertIsNotNone(await self.db.user_for_session_token(self.token))

    async def test_disabling_the_account_stops_every_session(self) -> None:
        """An account usable until its cookies expire is not disabled."""
        await self.db.set_user_active(self.user.id, is_active=False)
        self.assertIsNone(await self.db.user_for_session_token(self.token))

    async def test_absolute_lifetime_is_a_week(self) -> None:
        created, expires = await self.db.fetchrow(
            "SELECT created_at, expires_at FROM sessions WHERE id = ?", self.session_id
        )
        self.assertEqual(expires - created, ABSOLUTE_LIFETIME)

    async def test_revoke_all_sessions(self) -> None:
        other = new_session_token()
        await self.db.create_session(self.user.id, other)
        self.assertEqual(await self.db.revoke_user_sessions(self.user.id), 2)
        self.assertIsNone(await self.db.user_for_session_token(self.token))
        self.assertIsNone(await self.db.user_for_session_token(other))

    async def test_expired_rows_can_be_cleaned_up(self) -> None:
        await self.db.execute(
            "UPDATE sessions SET expires_at = ? WHERE id = ?",
            self.db._now() - timedelta(days=1),  # noqa: SLF001
            self.session_id,
        )
        self.assertEqual(await self.db.delete_expired_sessions(), 1)
        self.assertEqual(await self.db.fetchval("SELECT count(*) FROM sessions"), 0)

    async def test_connecting_address_is_recorded(self) -> None:
        # host() rather than ::text: inet carries a netmask, so a cast yields
        # "172.20.42.50/32".
        stored = await self.db.fetchval(
            "SELECT host(ip_address) FROM sessions WHERE id = ?", self.session_id
        )
        self.assertEqual(stored, "172.20.42.50")


class TestAuditLog(PostgresTestCase):
    async def test_attributed_event(self) -> None:
        user = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        await self.db.record_audit(
            "login_success",
            actor_user_id=user.id,
            actor_address="172.20.42.50",
            payload={"method": "google"},
        )
        rows = await self.db.recent_audit()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actor_email"], MINE)
        self.assertEqual(rows[0]["action_type"], "login_success")

    async def test_a_refused_login_is_recorded_without_a_user(self) -> None:
        """The event with no account to attribute is the one most worth keeping."""
        await self.db.record_audit(
            "login_denied",
            actor_address="2.39.73.4",
            payload={"email": "stranger@example.com", "reason": "not_allowlisted"},
        )
        rows = await self.db.recent_audit()
        self.assertIsNone(rows[0]["actor_user_id"])
        self.assertEqual(str(rows[0]["actor_address"]), "2.39.73.4")

    async def test_deleting_the_user_keeps_the_audit_trail(self) -> None:
        user = await self.db.record_login(SUB, MINE, "Iacopo", "admin")
        await self.db.record_audit("subject_renamed", actor_user_id=user.id, entity_id="42")
        await self.db.execute("DELETE FROM users WHERE id = ?", user.id)
        rows = await self.db.recent_audit()
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["actor_user_id"])

    async def test_entity_id_holds_a_non_uuid(self) -> None:
        """Cluster ids are integers, so one text column has to hold both."""
        await self.db.record_audit("cluster_dismissed", entity_type="cluster", entity_id="12345")
        self.assertEqual((await self.db.recent_audit())[0]["entity_id"], "12345")

    async def test_most_recent_first(self) -> None:
        for n in range(3):
            await self.db.record_audit(f"action_{n}")
        rows = await self.db.recent_audit()
        self.assertEqual([r["action_type"] for r in rows], ["action_2", "action_1", "action_0"])
