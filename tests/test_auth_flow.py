"""End-to-end tests for the login flow and the request gate (W5).

Drives the real routes against a real database. Google itself is the only
thing stubbed, at the one function that talks to it.
"""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ritrova.app import oidc, sessions
from ritrova.app.oidc import GoogleIdentity, OidcError

from ._postgres import PostgresAppTestCase

MINE = "iacopo.papalini@example.com"
STRANGER = "stranger@example.com"
SUB = "109876543210987654321"

ENV = {
    oidc.CLIENT_ID_VAR: "test-client.apps.googleusercontent.com",
    oidc.CLIENT_SECRET_VAR: "test-secret",
    oidc.PUBLIC_ORIGIN_VAR: "http://127.0.0.1:8787",
}


class TestGateDeniesByDefault(PostgresAppTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        sessions.clear_pending()
        self.client = await self.make_client(authenticated=False)

    def test_a_page_redirects_to_login(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn("/auth/login", response.headers["location"])

    def test_an_api_call_gets_json_not_html(self) -> None:
        """A redirect to an HTML page would surface as a parse error."""
        response = self.client.get("/api/export", follow_redirects=False)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "authentication required")

    def test_the_requested_page_is_remembered(self) -> None:
        response = self.client.get("/people/clusters", follow_redirects=False)
        self.assertIn("next=/people/clusters", response.headers["location"])

    def test_healthz_is_exempt(self) -> None:
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    def test_static_files_are_exempt(self) -> None:
        self.assertNotEqual(self.client.get("/static/style.css").status_code, 303)

    def test_a_write_is_refused(self) -> None:
        response = self.client.post("/api/subjects", json={"name": "x"}, follow_redirects=False)
        self.assertIn(response.status_code, (401, 403))


class TestLoginStart(PostgresAppTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        sessions.clear_pending()
        self._env = patch.dict("os.environ", ENV)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.client = await self.make_client(authenticated=False)

    def test_redirects_to_google(self) -> None:
        response = self.client.get("/auth/login", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers["location"].startswith(oidc.AUTH_ENDPOINT))

    def test_sets_the_state_cookie_that_binds_the_browser(self) -> None:
        response = self.client.get("/auth/login", follow_redirects=False)
        state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
        self.assertEqual(self.client.cookies.get(sessions.LOGIN_STATE_COOKIE), state)

    def test_an_open_redirect_is_not_honoured(self) -> None:
        """next= must never leave this site."""
        for hostile in ("https://evil.example.com/", "//evil.example.com", "javascript:alert(1)"):
            with self.subTest(target=hostile):
                self.assertEqual(_safe(hostile), "/")

    def test_without_credentials_it_says_so(self) -> None:
        with patch.dict("os.environ", {oidc.CLIENT_SECRET_VAR: ""}):
            response = self.client.get("/auth/login", follow_redirects=False)
        self.assertEqual(response.status_code, 503)


def _safe(target: str) -> str:
    from ritrova.app.routers.auth import _safe_next

    return _safe_next(target)


class TestCallback(PostgresAppTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        sessions.clear_pending()
        self._env = patch.dict("os.environ", ENV)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.client = await self.make_client(authenticated=False)

    def _start_login(self, next_url: str = "/") -> str:
        response = self.client.get(f"/auth/login?next={next_url}", follow_redirects=False)
        return parse_qs(urlparse(response.headers["location"]).query)["state"][0]

    async def test_an_allowlisted_user_gets_a_session(self) -> None:
        await self.db.allow_user(MINE, role="editor")
        state = self._start_login("/people/clusters")
        with patch.object(oidc, "exchange_code", return_value=GoogleIdentity(SUB, MINE, "Iacopo")):
            response = self.client.get(
                f"/auth/callback?code=the-code&state={state}", follow_redirects=False
            )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/people/clusters")
        self.assertTrue(self.client.cookies.get(sessions.SESSION_COOKIE))
        self.assertTrue(self.client.cookies.get(sessions.CSRF_COOKIE))

        user = await self.db.find_user_by_google_sub(SUB)
        assert user is not None
        self.assertEqual(user.role, "editor")
        actions = [r["action_type"] for r in await self.db.recent_audit()]
        self.assertIn("login_success", actions)

    async def test_the_new_session_actually_opens_pages(self) -> None:
        await self.db.allow_user(MINE)
        state = self._start_login()
        with patch.object(oidc, "exchange_code", return_value=GoogleIdentity(SUB, MINE, "Iacopo")):
            self.client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)
        self.assertEqual(self.client.get("/", follow_redirects=False).status_code, 200)

    async def test_a_stranger_is_refused_and_recorded(self) -> None:
        state = self._start_login()
        with patch.object(
            oidc, "exchange_code", return_value=GoogleIdentity("other-sub", STRANGER, None)
        ):
            response = self.client.get(
                f"/auth/callback?code=c&state={state}", follow_redirects=False
            )
        self.assertEqual(response.status_code, 403)
        self.assertIn("not on Ritrova", response.text)
        self.assertFalse(self.client.cookies.get(sessions.SESSION_COOKIE))

        rows = await self.db.recent_audit()
        self.assertEqual(rows[0]["action_type"], "login_denied")
        self.assertIsNone(rows[0]["actor_user_id"])
        self.assertEqual(rows[0]["payload"]["email"], STRANGER)

    async def test_a_mismatched_state_is_refused(self) -> None:
        """The login-CSRF case: a callback the browser never started."""
        self._start_login()
        response = self.client.get(
            "/auth/callback?code=c&state=a-state-from-somewhere-else", follow_redirects=False
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.client.cookies.get(sessions.SESSION_COOKIE))

    async def test_a_state_cannot_be_used_twice(self) -> None:
        await self.db.allow_user(MINE)
        state = self._start_login()
        with patch.object(oidc, "exchange_code", return_value=GoogleIdentity(SUB, MINE, None)):
            first = self.client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)
            self.assertEqual(first.status_code, 303)
            self.client.cookies.set(sessions.LOGIN_STATE_COOKIE, state)
            second = self.client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)
        self.assertEqual(second.status_code, 403)

    async def test_a_failed_exchange_is_refused_and_recorded(self) -> None:
        state = self._start_login()
        with patch.object(oidc, "exchange_code", side_effect=OidcError("bad nonce")):
            response = self.client.get(
                f"/auth/callback?code=c&state={state}", follow_redirects=False
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual((await self.db.recent_audit())[0]["action_type"], "login_failed")

    async def test_google_declining_is_recorded(self) -> None:
        self._start_login()
        response = self.client.get("/auth/callback?error=access_denied", follow_redirects=False)
        self.assertEqual(response.status_code, 403)
        self.assertEqual((await self.db.recent_audit())[0]["action_type"], "login_failed")

    async def test_a_disabled_account_is_refused(self) -> None:
        await self.db.allow_user(MINE)
        user = await self.db.record_login(SUB, MINE, None, "editor")
        await self.db.set_user_active(user.id, is_active=False)
        state = self._start_login()
        with patch.object(oidc, "exchange_code", return_value=GoogleIdentity(SUB, MINE, None)):
            response = self.client.get(
                f"/auth/callback?code=c&state={state}", follow_redirects=False
            )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.client.cookies.get(sessions.SESSION_COOKIE))


class TestCsrfAndRoles(PostgresAppTestCase):
    async def test_a_write_without_the_csrf_header_is_refused(self) -> None:
        client = await self.make_client()
        del client.headers[sessions.CSRF_HEADER]
        response = client.post("/api/subjects", json={"name": "Nobody"}, follow_redirects=False)
        self.assertEqual(response.status_code, 403)
        self.assertIn("CSRF", response.json()["error"])

    async def test_a_mismatched_csrf_header_is_refused(self) -> None:
        client = await self.make_client()
        client.headers[sessions.CSRF_HEADER] = "not-the-cookie-value"
        response = client.post("/api/subjects", json={"name": "Nobody"}, follow_redirects=False)
        self.assertEqual(response.status_code, 403)

    async def test_a_viewer_cannot_write(self) -> None:
        client = await self.make_client(role="viewer")
        response = client.post("/api/subjects", json={"name": "Nobody"}, follow_redirects=False)
        self.assertEqual(response.status_code, 403)
        self.assertIn("read-only", response.json()["error"])

    async def test_a_viewer_can_read(self) -> None:
        client = await self.make_client(role="viewer")
        self.assertEqual(client.get("/", follow_redirects=False).status_code, 200)

    async def test_reads_need_no_csrf_token(self) -> None:
        client = await self.make_client()
        del client.headers[sessions.CSRF_HEADER]
        self.assertEqual(client.get("/", follow_redirects=False).status_code, 200)


class TestLogout(PostgresAppTestCase):
    async def test_logout_revokes_the_session(self) -> None:
        client = await self.make_client()
        token = client.cookies.get(sessions.SESSION_COOKIE)
        assert token is not None
        self.assertIsNotNone(await self.db.user_for_session_token(token))

        response = client.post("/auth/logout", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIsNone(await self.db.user_for_session_token(token))

    async def test_logout_is_recorded(self) -> None:
        client = await self.make_client()
        client.post("/auth/logout", follow_redirects=False)
        self.assertIn("logout", [r["action_type"] for r in await self.db.recent_audit()])

    async def test_after_logout_pages_require_signing_in_again(self) -> None:
        client = await self.make_client()
        client.post("/auth/logout", follow_redirects=False)
        client.cookies.clear()
        self.assertEqual(client.get("/", follow_redirects=False).status_code, 303)
