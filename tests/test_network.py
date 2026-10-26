"""Tests for the LAN exemption (D18).

Drives the peer address directly, so none of this needs a NAS, a router
or a proxy. The forged-header cases are the regression tests that matter:
they are what fails if someone ever reads ``X-Forwarded-For`` here.
"""

from unittest import TestCase

from starlette.requests import Request

from ritrova.app.network import (
    RELAY_ADDRESSES,
    is_local_address,
    is_local_request,
    is_loopback_bind,
    peer_address,
)

LAN_HOST = "172.20.42.50"
NAS = "172.20.42.2"
ROUTER = "172.20.42.1"
PUBLIC = "2.39.73.4"


def _request(client_host: str | None, **headers: str) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 54321) if client_host else None,
        "server": ("172.20.42.3", 8787),
    }
    return Request(scope)


class TestIsLocalAddress(TestCase):
    def test_lan_host_is_local(self) -> None:
        self.assertTrue(is_local_address(LAN_HOST))

    def test_nas_is_not_local(self) -> None:
        """The relay exclusion: it holds a private address and is still not local."""
        self.assertFalse(is_local_address(NAS))

    def test_router_is_not_local(self) -> None:
        self.assertFalse(is_local_address(ROUTER))

    def test_public_address_is_not_local(self) -> None:
        self.assertFalse(is_local_address(PUBLIC))

    def test_loopback_is_local(self) -> None:
        self.assertTrue(is_local_address("127.0.0.1"))

    def test_missing_address_is_not_local(self) -> None:
        self.assertFalse(is_local_address(None))
        self.assertFalse(is_local_address(""))

    def test_unparseable_address_is_not_local(self) -> None:
        """Fail safe: if we cannot read it, we cannot claim it is local."""
        self.assertFalse(is_local_address("not-an-address"))
        self.assertFalse(is_local_address("172.20.42.50%eth0"))

    def test_ipv6(self) -> None:
        self.assertTrue(is_local_address("fd00::1"))
        self.assertTrue(is_local_address("::1"))
        self.assertFalse(is_local_address("2001:4860:4860::8888"))

    def test_every_relay_address_is_refused(self) -> None:
        """Whatever the list grows to, no member of it is ever local."""
        for address in RELAY_ADDRESSES:
            with self.subTest(address=address):
                self.assertFalse(is_local_address(address))


class TestIsLocalRequest(TestCase):
    def test_lan_request_is_local(self) -> None:
        self.assertTrue(is_local_request(_request(LAN_HOST)))

    def test_public_request_is_not_local(self) -> None:
        self.assertFalse(is_local_request(_request(PUBLIC)))

    def test_nas_request_is_not_local(self) -> None:
        self.assertFalse(is_local_request(_request(NAS)))

    def test_clientless_request_is_not_local(self) -> None:
        self.assertFalse(is_local_request(_request(None)))

    def test_forged_forwarded_for_cannot_make_an_outsider_local(self) -> None:
        """The attack this module exists to refuse."""
        request = _request(PUBLIC, x_forwarded_for=LAN_HOST)
        self.assertFalse(is_local_request(request))

    def test_forged_forwarded_for_cannot_make_a_lan_host_remote(self) -> None:
        request = _request(LAN_HOST, x_forwarded_for=PUBLIC)
        self.assertTrue(is_local_request(request))

    def test_forged_forwarded_for_cannot_launder_a_relay(self) -> None:
        """A request arriving from the NAS stays non-local whatever it claims."""
        request = _request(NAS, x_forwarded_for=LAN_HOST)
        self.assertFalse(is_local_request(request))

    def test_peer_address_ignores_the_header(self) -> None:
        request = _request(LAN_HOST, x_forwarded_for=PUBLIC, x_real_ip=PUBLIC)
        self.assertEqual(peer_address(request), LAN_HOST)


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
