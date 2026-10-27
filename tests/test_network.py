"""Tests for the address helpers behind the serving guards.

The LAN exemption these once supported was rejected; what remains decides
whether a bind is loopback-only, whether an unauthenticated process may
serve at all, and what address is safe to write to an ``inet`` column.
"""

from unittest import TestCase

from ritrova.app.network import (
    is_loopback_bind,
    refuse_unauthenticated_exposure,
    storable_address,
)

LAN_HOST = "172.20.42.50"
NAS = "172.20.42.2"
PUBLIC = "2.39.73.4"


class TestIsLoopbackBind(TestCase):
    def test_loopback_forms(self) -> None:
        for host in ("127.0.0.1", "127.0.0.53", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_bind(host))

    def test_every_interface_is_not_loopback(self) -> None:
        """The case the D19 guard exists to catch."""
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                self.assertFalse(is_loopback_bind(host))

    def test_routable_addresses_are_not_loopback(self) -> None:
        for host in ("172.20.42.3", LAN_HOST, PUBLIC):
            with self.subTest(host=host):
                self.assertFalse(is_loopback_bind(host))

    def test_unresolvable_host_is_not_loopback(self) -> None:
        self.assertFalse(is_loopback_bind(""))
        self.assertFalse(is_loopback_bind("paiolo.local"))


class TestRefuseUnauthenticatedExposure(TestCase):
    def test_loopback_without_auth_is_allowed(self) -> None:
        """Reachable only from this machine, so a login buys nothing."""
        self.assertIsNone(refuse_unauthenticated_exposure("127.0.0.1", auth_configured=False))
        self.assertIsNone(refuse_unauthenticated_exposure("localhost", auth_configured=False))

    def test_public_bind_without_auth_is_refused(self) -> None:
        for host in ("0.0.0.0", "::", "172.20.42.3", PUBLIC):
            with self.subTest(host=host):
                reason = refuse_unauthenticated_exposure(host, auth_configured=False)
                assert reason is not None
                self.assertIn("Refusing to serve", reason)
                self.assertIn(host, reason)

    def test_public_bind_with_auth_is_allowed(self) -> None:
        self.assertIsNone(refuse_unauthenticated_exposure("0.0.0.0", auth_configured=True))

    def test_the_reason_names_both_ways_out(self) -> None:
        reason = refuse_unauthenticated_exposure("0.0.0.0", auth_configured=False)
        assert reason is not None
        self.assertIn("RITROVA_GOOGLE_CLIENT_ID", reason)
        self.assertIn("--logged-user", reason)


class TestStorableAddress(TestCase):
    def test_passes_real_addresses_through(self) -> None:
        for value in (LAN_HOST, NAS, PUBLIC, "127.0.0.1", "fd00::1"):
            with self.subTest(value=value):
                self.assertEqual(storable_address(value), value)

    def test_drops_anything_that_is_not_an_address(self) -> None:
        """An inet column would raise, and an audit write that raises would
        fail the very request it was recording."""
        for value in ("testclient", "", None, "paiolo.local", "172.20.42.50%eth0"):
            with self.subTest(value=value):
                self.assertIsNone(storable_address(value))
