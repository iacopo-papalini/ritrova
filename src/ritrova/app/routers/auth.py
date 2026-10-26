"""Google sign-in, sign-out and the pages that explain a refusal (W5).

The authorization-code flow: ``/auth/login`` sends the browser to Google,
``/auth/callback`` turns the code it comes back with into a local session.
Both are exempt from the request gate, since a login cannot require being
logged in.

Registered **before** the ``pages`` router, like every other non-page
router: ``pages`` owns ``/{kind}/{subject_id}``, a two-segment catch-all
that would otherwise swallow ``/auth/callback`` as
``kind="auth", subject_id="callback"``. See CLAUDE.md:16.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ...db.auth import new_session_token
from .. import oidc, sessions
from ..deps import get_db, get_templates
from ..network import peer_address, storable_address

router = APIRouter(prefix="/auth", tags=["auth"])


def _safe_next(target: str | None) -> str:
    """Reduce ``next`` to a local path.

    Anything absolute, scheme-relative or not rooted is discarded: echoing
    it back in a redirect would make this endpoint an open redirect, which
    is the classic way a login page is turned into a phishing hop.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


@router.get("/login")
async def login(request: Request, next: str = Query(default="/")) -> Response:
    """Begin a Google login, or go straight through if already signed in."""
    destination = _safe_next(next)

    if sessions.current_user(request) is not None:
        return RedirectResponse(destination, status_code=303)

    if not oidc.google_oidc_configured():
        return get_templates().TemplateResponse(
            name="auth_unavailable.html",
            context={},
            request=request,
            status_code=503,
        )

    state = oidc.new_state()
    nonce = oidc.new_nonce()
    verifier = oidc.new_code_verifier()
    sessions.remember_login(
        state,
        sessions.PendingLogin(
            nonce=nonce,
            code_verifier=verifier,
            started_at=time.monotonic(),
            next_url=destination,
        ),
    )
    response = RedirectResponse(
        oidc.authorization_url(state=state, nonce=nonce, code_verifier=verifier),
        status_code=303,
    )
    sessions.set_login_state_cookie(response, state)
    return response


@router.get("/callback")
async def callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> Response:
    """Turn Google's authorization code into a local session."""
    db = get_db()
    address = storable_address(peer_address(request))

    def refuse(reason: str) -> HTMLResponse:
        return _refused(request, reason=reason, status=403)

    if error is not None:
        await db.record_audit(
            "login_failed", actor_address=address, payload={"google_error": error}
        )
        return refuse("google_declined")

    # The state cookie is what binds this callback to the browser that
    # started the login. Without it an attacker could start a login and hand
    # the victim the callback URL, leaving the victim signed in as them.
    cookie_state = request.cookies.get(sessions.LOGIN_STATE_COOKIE)
    if not state or not cookie_state or state != cookie_state:
        await db.record_audit("login_failed", actor_address=address, payload={"reason": "state"})
        return refuse("state_mismatch")

    pending = sessions.take_login(state)
    if pending is None or not code:
        await db.record_audit("login_failed", actor_address=address, payload={"reason": "expired"})
        return refuse("login_expired")

    try:
        identity = await oidc.exchange_code(
            code, code_verifier=pending.code_verifier, expected_nonce=pending.nonce
        )
    except oidc.OidcError as exc:
        await db.record_audit("login_failed", actor_address=address, payload={"reason": str(exc)})
        return refuse("exchange_failed")

    invited = await db.find_allowed_user(identity.email)
    if invited is None:
        # Recorded with the address and no user, which is exactly the case
        # audit_log.actor_user_id is nullable for.
        await db.record_audit(
            "login_denied",
            actor_address=address,
            payload={"email": identity.email, "reason": "not_allowlisted"},
        )
        return _refused(request, reason="not_allowlisted", status=403, email=identity.email)

    user = await db.record_login(
        identity.google_sub, identity.email, identity.display_name, invited.role
    )
    if not user.is_active:
        await db.record_audit(
            "login_denied",
            actor_user_id=user.id,
            actor_address=address,
            payload={"reason": "disabled"},
        )
        return _refused(request, reason="account_disabled", status=403, email=identity.email)

    token = new_session_token()
    session_id = await db.create_session(
        user.id,
        token,
        user_agent=request.headers.get("user-agent"),
        ip_address=address,
    )
    await db.record_audit(
        "login_success",
        actor_user_id=user.id,
        actor_address=address,
        entity_type="session",
        entity_id=str(session_id),
    )

    response = RedirectResponse(pending.next_url, status_code=303)
    sessions.set_session_cookie(response, token)
    sessions.set_csrf_cookie(response, sessions.new_csrf_token())
    sessions.clear_login_state_cookie(response)
    return response


@router.post("/logout")
@router.get("/logout")
async def logout(request: Request) -> Response:
    """End this session. Google's own session is deliberately left alone."""
    db = get_db()
    user = sessions.current_user(request)
    token = request.cookies.get(sessions.SESSION_COOKIE)
    if token:
        found = await db.user_for_session_token(token)
        if found is not None:
            await db.revoke_session(found[1])
    if user is not None:
        await db.record_audit(
            "logout", actor_user_id=user.id, actor_address=storable_address(peer_address(request))
        )
    response = RedirectResponse("/auth/signed-out", status_code=303)
    sessions.clear_session_cookie(response)
    return response


@router.get("/signed-out")
async def signed_out(request: Request) -> Response:
    return get_templates().TemplateResponse(
        name="auth_signed_out.html", context={}, request=request
    )


def _refused(
    request: Request, *, reason: str, status: int, email: str | None = None
) -> HTMLResponse:
    return get_templates().TemplateResponse(
        name="auth_refused.html",
        context={"reason": reason, "email": email},
        request=request,
        status_code=status,
    )
