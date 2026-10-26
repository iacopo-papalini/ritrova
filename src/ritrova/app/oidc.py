"""Google OpenID Connect: configuration, authorization URL, code exchange (W5).

The authorization-code flow with PKCE. Ritrova exchanges the code once at
login and runs its own cookie session afterwards, so no Google token is ever
stored or refreshed — which is also why Google's Testing-mode 7-day
re-consent window, a limit on refresh tokens, does not apply here.

**On not verifying the ID token's signature.** The ID token is read for its
claims without checking its JWT signature, which is deliberate and
permitted: OpenID Connect Core 3.1.3.7 item 6 says that when the token is
received by direct communication with the token endpoint, "the TLS server
validation MAY be used to validate the issuer in place of checking the token
signature". This code fetches the token itself from Google's token endpoint
over certificate-verified HTTPS, so that condition holds. Every *claim* is
still validated below. If an ID token ever arrives by any other route — an
implicit flow, a client-side redirect, a header — this reasoning collapses
and the signature must be verified against Google's JWKS.

The credentials come from the environment, never from a tracked file, and
the secret is read on demand rather than held in a module global.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

CLIENT_ID_VAR = "RITROVA_GOOGLE_CLIENT_ID"
CLIENT_SECRET_VAR = "RITROVA_GOOGLE_CLIENT_SECRET"
PUBLIC_ORIGIN_VAR = "RITROVA_PUBLIC_ORIGIN"

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# Google issues both spellings and both are correct.
VALID_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

CALLBACK_PATH = "/auth/callback"

# Only these three. Google exempts an app requesting nothing further from
# verification, from the trusted-tester list, and from the 7-day re-consent
# cycle; any additional scope voids that, so this list is a thing to guard
# in review rather than to extend casually.
SCOPES = ("openid", "email", "profile")

# Allows for ordinary clock skew between here and Google without widening
# the window enough to matter.
_CLOCK_SKEW_SECONDS = 120

_TOKEN_TIMEOUT = httpx.Timeout(10.0)


class OidcError(Exception):
    """A login that cannot be completed. The message is safe to log, not to show."""


@dataclass(frozen=True)
class GoogleIdentity:
    google_sub: str
    email: str
    display_name: str | None


def client_id() -> str | None:
    return os.environ.get(CLIENT_ID_VAR) or None


def client_secret() -> str | None:
    return os.environ.get(CLIENT_SECRET_VAR) or None


def public_origin() -> str | None:
    origin = os.environ.get(PUBLIC_ORIGIN_VAR) or None
    return origin.rstrip("/") if origin else None


def google_oidc_configured() -> bool:
    """Whether both halves of the OAuth client are present.

    Both, because either alone cannot complete a code exchange — a half
    configuration would otherwise pass the startup check and then fail at
    the one moment a family member tries to log in.
    """
    return client_id() is not None and client_secret() is not None


def redirect_uri() -> str:
    """The callback URI, built from an explicit origin rather than the request.

    Never derived from the ``Host`` header: that header is supplied by the
    client (or by a proxy relaying one), so building a redirect URI from it
    is host-header injection with an authorization code as the prize. Google
    also compares this string exactly against the registered value, so one
    configured origin is easier to keep in step than a derived one.
    """
    origin = public_origin()
    if origin is None:
        msg = f"{PUBLIC_ORIGIN_VAR} is not set, so the callback URL cannot be built."
        raise OidcError(msg)
    return f"{origin}{CALLBACK_PATH}"


def new_state() -> str:
    return secrets.token_urlsafe(32)


def new_nonce() -> str:
    return secrets.token_urlsafe(32)


def new_code_verifier() -> str:
    return secrets.token_urlsafe(64)


def code_challenge_for(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def authorization_url(*, state: str, nonce: str, code_verifier: str) -> str:
    cid = client_id()
    if cid is None:
        msg = f"{CLIENT_ID_VAR} is not set."
        raise OidcError(msg)
    params = {
        "client_id": cid,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge_for(code_verifier),
        "code_challenge_method": "S256",
        # Ritrova asks Google who the visitor is and then runs its own
        # session, so there is nothing to keep offline access for.
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Read a JWT's claims without verifying its signature.

    See the module docstring: valid only because the caller fetched this
    token directly from the token endpoint over verified TLS.
    """
    parts = token.split(".")
    if len(parts) != 3:
        msg = "ID token is not a three-part JWT."
        raise OidcError(msg)
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError) as exc:
        msg = "ID token payload is not readable JSON."
        raise OidcError(msg) from exc
    if not isinstance(claims, dict):
        msg = "ID token payload is not a JSON object."
        raise OidcError(msg)
    return claims


def validate_claims(claims: dict[str, Any], *, expected_nonce: str) -> GoogleIdentity:
    """Check every claim this design relies on, then return the identity."""
    issuer = claims.get("iss")
    if issuer not in VALID_ISSUERS:
        msg = f"ID token issuer {issuer!r} is not Google."
        raise OidcError(msg)

    cid = client_id()
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if cid not in audiences:
        msg = "ID token was issued for a different OAuth client."
        raise OidcError(msg)

    # azp identifies the party the token was issued to, and differs from aud
    # when a token is passed between clients. Ours should never be relayed.
    authorized_party = claims.get("azp")
    if authorized_party is not None and authorized_party != cid:
        msg = "ID token was issued to a different authorized party."
        raise OidcError(msg)

    expiry = claims.get("exp")
    if not isinstance(expiry, int | float):
        msg = "ID token has no usable expiry."
        raise OidcError(msg)
    if time.time() - _CLOCK_SKEW_SECONDS > expiry:
        msg = "ID token has expired."
        raise OidcError(msg)

    if claims.get("nonce") != expected_nonce:
        msg = "ID token nonce does not match the login that started this."
        raise OidcError(msg)

    # An unverified address could be anyone's: Google only attests that the
    # account exists, not that its owner controls that mailbox. The allowlist
    # is keyed on the address, so accepting one unverified would let a
    # stranger claim a family member's invitation.
    if claims.get("email_verified") is not True:
        msg = "Google has not verified this account's email address."
        raise OidcError(msg)

    subject = claims.get("sub")
    email = claims.get("email")
    if not subject or not email:
        msg = "ID token is missing sub or email."
        raise OidcError(msg)

    return GoogleIdentity(
        google_sub=str(subject),
        email=str(email),
        display_name=claims.get("name") or None,
    )


async def exchange_code(code: str, *, code_verifier: str, expected_nonce: str) -> GoogleIdentity:
    """Trade an authorization code for a validated Google identity."""
    cid, secret = client_id(), client_secret()
    if cid is None or secret is None:
        msg = "Google sign-in is not configured."
        raise OidcError(msg)

    form = {
        "code": code,
        "client_id": cid,
        "client_secret": secret,
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }
    try:
        # verify=True is httpx's default and is what the unverified-signature
        # reasoning rests on; it must not be relaxed here.
        async with httpx.AsyncClient(timeout=_TOKEN_TIMEOUT) as client:
            response = await client.post(TOKEN_ENDPOINT, data=form)
    except httpx.HTTPError as exc:
        msg = f"Could not reach Google's token endpoint: {exc}"
        raise OidcError(msg) from exc

    if response.status_code != httpx.codes.OK:
        msg = f"Google refused the code exchange ({response.status_code})."
        raise OidcError(msg)

    try:
        body = response.json()
    except ValueError as exc:
        msg = "Google's token response was not JSON."
        raise OidcError(msg) from exc

    id_token = body.get("id_token")
    if not id_token:
        msg = "Google's token response carried no id_token."
        raise OidcError(msg)

    return validate_claims(_decode_jwt_payload(id_token), expected_nonce=expected_nonce)
