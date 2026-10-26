"""Deciding whether a request came from inside the house.

Reads are open to LAN clients and every write is authenticated (D18), so
this module answers one question: is the peer on our own network?

**The answer comes from the connection's own address, never from a
header.** ``X-Forwarded-For`` is attacker-controlled on any request that
did not pass through our proxy, and uvicorn only overwrites
``request.client`` from it when the peer is in ``--forwarded-allow-ips``.
Reading the header directly would let a LAN client claim to be anyone, or
an outside client claim to be local.

The exemption carries one further rule, and it is the reason this is a
module rather than an ``ip.is_private`` call at the call site:

    Exempt only hosts that cannot relay traffic from outside the network.

TLS terminates on the NAS, which proxies to this app, so a visitor from
the internet arrives over a connection whose peer address is the NAS — a
private address. Exempting every private address would therefore exempt
the whole internet while behaving perfectly from the LAN, which is where
it would be tested. Excluding the relays makes that misconfiguration
fail safe: proxied visitors are asked to log in, which is wrong but
visible, rather than silently admitted.

``RELAY_ADDRESSES`` is a list to be added to, not two special cases: any
host that gains the ability to carry traffic in from outside belongs in
it — another reverse proxy, a VPN concentrator, a container host doing
its own NAT.
"""

from __future__ import annotations

import ipaddress

from starlette.requests import Request

# Hosts on our network that can relay traffic from outside it, and so must
# never be treated as local however private their address looks. The NAS
# terminates TLS and proxies inward; the router could be made to forward a
# port straight here, and whether it then rewrites the source address is
# firmware behaviour we would rather not depend on.
RELAY_ADDRESSES = frozenset(
    {
        "172.20.42.2",  # NAS — reverse proxy, TLS termination
        "172.20.42.1",  # router — a port forward would arrive through it
    }
)


def is_loopback_bind(host: str) -> bool:
    """Whether binding ``host`` exposes the app to this machine only.

    Running without authentication is legitimate on a laptop and unsafe
    anywhere else, so the bind address is what decides whether it is
    allowed (D19). ``0.0.0.0`` and ``::`` are every interface, not
    loopback, and are the cases this exists to catch.
    """
    if host in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # An empty host, a name we cannot resolve here, or a malformed
        # address: none of them demonstrate loopback, so none of them pass.
        return False


def refuse_unauthenticated_exposure(host: str, *, auth_configured: bool) -> str | None:
    """Why serving on ``host`` without authentication must be refused, or None.

    Returns the reason rather than raising, so the CLI can present it and
    tests can assert on it without catching.

    A bind beyond loopback is reachable by anything that can route to this
    machine — which, behind the NAS's reverse proxy, is the internet. There
    is no configuration of this app that makes that safe without a login,
    so it is refused at startup instead of trusting whoever deploys it to
    notice.
    """
    if is_loopback_bind(host):
        return None
    if auth_configured:
        return None
    return (
        f"Refusing to serve on {host!r} without authentication. Anything that can "
        "route to this machine could read and change the archive, and behind a "
        "reverse proxy that means the internet.\n"
        "Either configure Google sign-in (RITROVA_GOOGLE_CLIENT_ID and "
        "RITROVA_GOOGLE_CLIENT_SECRET), or bind loopback with "
        "--host 127.0.0.1 --logged-user <allowlisted address>."
    )


def storable_address(value: str | None) -> str | None:
    """``value`` if it is an IP address, else None.

    ``request.client.host`` is not guaranteed to be one: a UNIX-socket peer,
    a test client and a malformed proxy header all produce something else.
    The audit and session columns are ``inet``, which rejects those, and an
    audit write that raises would fail the request it was recording — so a
    login would break over a value we only wanted for the record. Dropping
    the address is the lesser loss, and the row still carries everything
    else.
    """
    if not value:
        return None
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return None
    return value


def is_local_address(peer_address: str | None) -> bool:
    """Whether ``peer_address`` is a host on our network that is not a relay.

    Takes the address rather than a request so the decision can be driven
    directly in tests, with no proxy, NAS or router involved.
    """
    if not peer_address:
        return False
    try:
        address = ipaddress.ip_address(peer_address)
    except ValueError:
        # Unparseable means we cannot show it is local, so it is not.
        return False
    if not address.is_private:
        return False
    return peer_address not in RELAY_ADDRESSES


def peer_address(request: Request) -> str | None:
    """The address Starlette reports for the connection itself.

    Already reflects ``X-Forwarded-For`` when uvicorn was told to trust
    the proxy it came from, and ignores the header otherwise. Either way
    no header is consulted here.
    """
    return request.client.host if request.client else None


def is_local_request(request: Request) -> bool:
    return is_local_address(peer_address(request))
