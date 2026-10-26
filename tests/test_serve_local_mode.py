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
    def _invoke(self, *args: str, logged_user: str | None = None):
        # ``cli.py`` calls load_dotenv() at import, so a developer's own .env
        # would otherwise decide what these tests see. The variable is set
        # explicitly here, never inherited.
        env = {"RITROVA_LOGGED_USER": logged_user}
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

    def test_without_logged_user_any_bind_is_fine(self) -> None:
        """The guard constrains local mode only; a normal server still binds wide."""
        result, run, _ = self._invoke("--host", "0.0.0.0")
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once()

    def test_identity_is_passed_to_the_app(self) -> None:
        _, _, create_app = self._invoke("--host", "127.0.0.1", "--logged-user", MINE)
        self.assertEqual(create_app.call_args.kwargs["logged_user"], MINE)

    def test_env_var_supplies_the_address(self) -> None:
        """The address lives in .env, which is gitignored, not in the Makefile."""
        result, run, create_app = self._invoke("--host", "0.0.0.0", logged_user=MINE)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("only allowed when bound to loopback", result.output)
        run.assert_not_called()
        create_app.assert_not_called()
