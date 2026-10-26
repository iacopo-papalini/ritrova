"""Tests for local mode (D19): --logged-user is loopback-only.

Local mode stands in for a finished Google login, so anyone who can reach
the port is the named user. The bind address is therefore what decides
whether it is allowed, and the refusal has to be an error — a warning on a
server nobody is watching is an unauthenticated Ritrova on the network.

These assert on the refusal, never on a started server: the guard runs
before uvicorn, so no socket is opened.
"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from click.testing import CliRunner

from ritrova.cli import cli

MINE = "iacopo.papalini@example.com"


class TestLoggedUserRequiresLoopback(TestCase):
    def _invoke(self, *args: str, logged_user: str | None = None, auth: bool = False):  # noqa: C901
        # ``cli.py`` calls load_dotenv() at import, so a developer's own .env
        # would otherwise decide what these tests see. Every variable the
        # guards read is set explicitly here, never inherited.
        env = {
            "RITROVA_GOOGLE_CLIENT_ID": "test-client-id" if auth else None,
            "RITROVA_GOOGLE_CLIENT_SECRET": "test-client-secret" if auth else None,
        }
        if logged_user is not None:
            args = (*args, "--logged-user", logged_user)
        # Patched so a refusal is proved by uvicorn never being reached,
        # rather than by a port that may or may not bind on CI.
        with patch("uvicorn.run") as run, patch("ritrova.app.create_app") as create_app:
            result = CliRunner().invoke(cli, ["serve", *args], env=env)
        return result, run, create_app

    def test_refused_on_every_interface(self) -> None:
        result, run, _ = self._invoke("--host", "0.0.0.0", "--logged-user", MINE)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("only allowed when bound to loopback", result.output)
        run.assert_not_called()

    def test_refused_on_a_lan_address(self) -> None:
        result, run, _ = self._invoke("--host", "172.20.42.3", "--logged-user", MINE)
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()

    def test_allowed_on_loopback(self) -> None:
        result, run, _ = self._invoke("--host", "127.0.0.1", "--logged-user", MINE)
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once()

    def test_allowed_on_localhost(self) -> None:
        _, run, _ = self._invoke("--host", "localhost", "--logged-user", MINE)
        run.assert_called_once()

    def test_wide_bind_needs_auth_even_without_logged_user(self) -> None:
        """Dropping --logged-user is not a way to serve the LAN unauthenticated."""
        result, run, _ = self._invoke("--host", "0.0.0.0")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Refusing to serve", result.output)
        run.assert_not_called()

    def test_wide_bind_is_allowed_once_sign_in_is_configured(self) -> None:
        result, run, _ = self._invoke("--host", "0.0.0.0", auth=True)
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once()

    def test_identity_is_passed_to_the_app(self) -> None:
        _, _, create_app = self._invoke("--host", "127.0.0.1", "--logged-user", MINE)
        self.assertEqual(create_app.call_args.kwargs["logged_user"], MINE)

    def test_the_environment_cannot_enter_local_mode(self) -> None:
        """Only the explicit flag may. A value in .env would apply silently to
        every run from that directory and produce a server that looks
        authenticated and is not."""
        with patch("uvicorn.run") as run, patch("ritrova.app.create_app") as create_app:
            result = CliRunner().invoke(
                cli,
                ["serve", "--host", "127.0.0.1"],
                env={
                    "RITROVA_LOGGED_USER": MINE,
                    "RITROVA_GOOGLE_CLIENT_ID": None,
                    "RITROVA_GOOGLE_CLIENT_SECRET": None,
                },
            )
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once()
        self.assertIsNone(create_app.call_args.kwargs["logged_user"])
        self.assertNotIn("Signed in as", result.output)


class TestPaioloCannotServeWithoutAuth(TestCase):
    """The deployment shape: paiolo binds 0.0.0.0 behind the NAS proxy."""

    def _serve(self, env: dict[str, str | None]):
        with patch("uvicorn.run") as run:
            result = CliRunner().invoke(cli, ["serve", "--host", "0.0.0.0"], env=env)
        return result, run

    def test_refused_with_no_credentials(self) -> None:
        result, run = self._serve(
            {
                "RITROVA_LOGGED_USER": None,
                "RITROVA_GOOGLE_CLIENT_ID": None,
                "RITROVA_GOOGLE_CLIENT_SECRET": None,
            }
        )
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()

    def test_refused_with_only_the_client_id(self) -> None:
        """Half a configuration would otherwise pass startup and fail at login."""
        result, run = self._serve(
            {
                "RITROVA_LOGGED_USER": None,
                "RITROVA_GOOGLE_CLIENT_ID": "test-client-id",
                "RITROVA_GOOGLE_CLIENT_SECRET": None,
            }
        )
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()

    def test_refused_with_only_the_secret(self) -> None:
        result, run = self._serve(
            {
                "RITROVA_LOGGED_USER": None,
                "RITROVA_GOOGLE_CLIENT_ID": None,
                "RITROVA_GOOGLE_CLIENT_SECRET": "test-client-secret",
            }
        )
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()

    def test_refused_when_a_variable_is_set_but_empty(self) -> None:
        result, run = self._serve(
            {
                "RITROVA_LOGGED_USER": None,
                "RITROVA_GOOGLE_CLIENT_ID": "",
                "RITROVA_GOOGLE_CLIENT_SECRET": "",
            }
        )
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()
