"""Address questions the serving decisions depend on.

Two of them, and they are unrelated:

``is_loopback_bind`` answers whether a bind address exposes the app to this
machine only, which is what decides whether ``--logged-user`` — the local
single-user mode used to feed new images on the Mac — is allowed, and
whether a process may serve at all without Google sign-in configured.

``peer_address`` / ``storable_address`` answer who is connecting, for the
audit log. **Never from a header.** ``X-Forwarded-For`` is
attacker-controlled on any request that did not pass through our proxy, and
uvicorn only overwrites ``request.client`` from it when the peer is in
``--forwarded-allow-ips`` — which the deployment sets to the NAS alone.

**A LAN exemption was considered and deliberately rejected** (superseding
D18, decided 2026-10-26). Every request is authenticated, wherever it comes
from. The implementation existed — a private-range test that excluded the
NAS and the router, because TLS terminates on the NAS and so a visitor from
the internet arrives wearing a private address — and it is gone because the
exemption it served is gone. Do not reinstate it on the strength of the
comment above: signing in costs one click, sessions last 12 hours idle, and
requiring one everywhere is what keeps a name against every row of the
audit log. The relay problem it solved cannot arise when nothing is exempt.
"""

from __future__ import annotations

import ipaddress

from starlette.requests import Request


def is_loopback_bind(host: str) -> bool:
    """Whether binding ``host`` exposes the app to this machine only.

    Running the local single-user mode is legitimate on a laptop and unsafe
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


def peer_address(request: Request) -> str | None:
    """The address Starlette reports for the connection itself.

    Already reflects ``X-Forwarded-For`` when uvicorn was told to trust the
    proxy it came from, and ignores the header otherwise. Either way no
    header is consulted here.
    """
    return request.client.host if request.client else None
