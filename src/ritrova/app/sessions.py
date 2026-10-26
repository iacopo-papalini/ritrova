"""Session cookies, CSRF and the request gate (W5).

Three things live here: what a session cookie looks like, the short-lived
state of a login in progress, and the middleware that decides whether a
request may proceed.

**The gate's default is deny.** Paths are exempted by an explicit list, so
a route added later is protected unless someone deliberately exempts it —
the opposite way round from checking for a session only where it was
remembered.

**Why a login in progress is held here rather than in a cookie.** The nonce
and the PKCE verifier stay server-side, keyed by the opaque ``state``; only
the state itself reaches the browser. The state is *also* set as a cookie
and compared at the callback, which is what stops login CSRF: without it an
attacker could start a login, hand the victim the resulting callback URL and
have the victim's browser adopt the attacker's identity.

Single process by design (D2), so a dict is the whole store. A restart
mid-login costs a retry.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from ..db.auth import User
from . import oidc

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

SESSION_COOKIE = "ritrova_session"
CSRF_COOKIE = "ritrova_csrf"
CSRF_HEADER = "X-CSRF-Token"
LOGIN_STATE_COOKIE = "ritrova_login"

# Long enough for a Google consent screen with a password and 2FA prompt,
# short enough that an abandoned login does not linger.
LOGIN_TIMEOUT_SECONDS = 600

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Reachable without a session. Everything else needs one.
#   /auth/*    — the login flow itself cannot require being logged in
#   /healthz   — the container's healthcheck has no cookie jar
#   /static/*  — stylesheets and scripts, with nothing of the archive in them
EXEMPT_PREFIXES = ("/auth/", "/static/")
EXEMPT_PATHS = frozenset({"/healthz", "/favicon.ico"})


@dataclass(frozen=True)
class PendingLogin:
    nonce: str
    code_verifier: str
    started_at: float
    next_url: str


_pending: dict[str, PendingLogin] = {}


def remember_login(state: str, pending: PendingLogin) -> None:
    _expire_pending()
    _pending[state] = pending


def take_login(state: str) -> PendingLogin | None:
    """Retrieve and remove a login in progress. One use only, by design."""
    _expire_pending()
    return _pending.pop(state, None)


def _expire_pending() -> None:
    cutoff = time.monotonic() - LOGIN_TIMEOUT_SECONDS
    for state in [s for s, p in _pending.items() if p.started_at < cutoff]:
        del _pending[state]


def clear_pending() -> None:
    """Drop every login in progress. For tests and for logout-everywhere."""
    _pending.clear()


def cookies_are_secure() -> bool:
    """Whether to mark cookies ``Secure``.

    Keyed on the configured origin's scheme rather than hardcoded: a
    ``Secure`` cookie is not sent over plain HTTP at all, so hardcoding it
    true would break the loopback development origin, and hardcoding it
    false would put the session cookie in clear on the family WiFi.
    """
    origin = oidc.public_origin()
    return origin is None or origin.startswith("https://")


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(_absolute_lifetime_seconds()),
        httponly=True,
        secure=cookies_are_secure(),
        samesite="lax",
        path="/",
    )


def _absolute_lifetime_seconds() -> float:
    from ..db.auth import ABSOLUTE_LIFETIME

    return ABSOLUTE_LIFETIME.total_seconds()


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: str) -> None:
    """Set the CSRF cookie, readable by JavaScript on purpose.

    This is the double-submit pattern: the browser's own script reads the
    cookie and echoes it in a header, which a cross-site page cannot do
    because it cannot read our cookies. So this one is not ``HttpOnly`` —
    the session cookie, which must never be script-readable, still is.
    """
    response.set_cookie(
        CSRF_COOKIE,
        token,
        max_age=int(_absolute_lifetime_seconds()),
        httponly=False,
        secure=cookies_are_secure(),
        samesite="lax",
        path="/",
    )


def set_login_state_cookie(response: Response, state: str) -> None:
    response.set_cookie(
        LOGIN_STATE_COOKIE,
        state,
        max_age=LOGIN_TIMEOUT_SECONDS,
        httponly=True,
        secure=cookies_are_secure(),
        samesite="lax",
        path="/",
    )


def clear_login_state_cookie(response: Response) -> None:
    response.delete_cookie(LOGIN_STATE_COOKIE, path="/")


def is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith(EXEMPT_PREFIXES)


def current_user(request: Request) -> User | None:
    return getattr(request.state, "user", None)


def csrf_token_matches(request: Request) -> bool:
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header:
        return False
    return secrets.compare_digest(cookie, header)


def _wants_json(request: Request) -> bool:
    """Whether to answer with JSON rather than a redirect to the login page.

    An ``/api`` call redirected to an HTML login page produces a parse error
    in the client rather than anything a user can act on, so these two have
    to be told apart.
    """
    return request.url.path.startswith("/api/") or "application/json" in request.headers.get(
        "accept", ""
    )


def unauthenticated_response(request: Request) -> Response:
    from .responses import JSONResponse

    if _wants_json(request):
        return JSONResponse({"error": "authentication required"}, status_code=401)
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(f"/auth/login?next={target}", status_code=303)


def install(app: object) -> None:
    """Attach the gate to a FastAPI app.

    Registered as middleware rather than per-route dependencies so that a
    route added later cannot miss it.
    """
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)

    @app.middleware("http")
    async def gate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        from . import deps
        from .responses import JSONResponse

        local_identity = getattr(request.app.state, "logged_user", None)
        request.state.user = None
        session_id = None

        if local_identity is not None:
            # Local mode: the allowlist entry stands in for a finished Google
            # login, resolved at startup. Still a real identity, so the audit
            # log and the role checks behave as they do in production.
            db = deps.get_db()
            request.state.user = await db.find_user_by_email(local_identity.email)
            if request.state.user is None:
                request.state.user = await db.record_login(
                    f"local-mode:{local_identity.email}",
                    local_identity.email,
                    None,
                    local_identity.role,
                )
        else:
            token = request.cookies.get(SESSION_COOKIE)
            if token:
                db = deps.get_db()
                found = await db.user_for_session_token(token)
                if found is not None:
                    request.state.user, session_id = found

        if not is_exempt(request.url.path) and request.state.user is None:
            return unauthenticated_response(request)

        if request.method in MUTATING_METHODS and not is_exempt(request.url.path):
            if not csrf_token_matches(request):
                return JSONResponse({"error": "CSRF token missing or invalid"}, status_code=403)
            user = request.state.user
            if user is not None and not user.may_write:
                return JSONResponse({"error": "your role is read-only"}, status_code=403)

        response = await call_next(request)

        if session_id is not None:
            await deps.get_db().touch_session(session_id)
        return response
