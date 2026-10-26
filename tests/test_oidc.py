"""Tests for the Google OIDC client (W5).

``validate_claims`` is the security boundary of the whole login flow: every
case below is a token that must be refused, and each corresponds to a way a
stranger could otherwise end up holding a Ritrova session.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ritrova.app import oidc
from ritrova.app.oidc import OidcError

CLIENT_ID = "test-client-id.apps.googleusercontent.com"
OTHER_CLIENT = "other-client-id.apps.googleusercontent.com"
NONCE = "the-nonce-from-this-login"
SUB = "109876543210987654321"
EMAIL = "iacopo.papalini@example.com"


def _claims(**overrides: Any) -> dict[str, Any]:
    base = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "azp": CLIENT_ID,
        "sub": SUB,
        "email": EMAIL,
        "email_verified": True,
        "name": "Iacopo Papalini",
        "exp": int(time.time()) + 3600,
        "nonce": NONCE,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not _ABSENT}


_ABSENT = object()


def _b64(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def _id_token(claims: dict[str, Any]) -> str:
    return f"{_b64({'alg': 'RS256'})}.{_b64(claims)}.signature-not-checked"


class OidcTestCase(TestCase):
    def setUp(self) -> None:
        self._env = patch.dict(
            "os.environ",
            {
                oidc.CLIENT_ID_VAR: CLIENT_ID,
                oidc.CLIENT_SECRET_VAR: "test-secret",
                oidc.PUBLIC_ORIGIN_VAR: "https://photos.papalini.casa",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)


class TestConfiguration(OidcTestCase):
    def test_configured_when_both_halves_present(self) -> None:
        self.assertTrue(oidc.google_oidc_configured())

    def test_not_configured_without_the_secret(self) -> None:
        with patch.dict("os.environ", {oidc.CLIENT_SECRET_VAR: ""}):
            self.assertFalse(oidc.google_oidc_configured())

    def test_redirect_uri_uses_the_configured_origin(self) -> None:
        self.assertEqual(oidc.redirect_uri(), "https://photos.papalini.casa/auth/callback")

    def test_trailing_slash_on_the_origin_is_tolerated(self) -> None:
        """Google matches the URI exactly, so a stray slash must not reach it."""
        with patch.dict("os.environ", {oidc.PUBLIC_ORIGIN_VAR: "https://photos.papalini.casa/"}):
            self.assertEqual(oidc.redirect_uri(), "https://photos.papalini.casa/auth/callback")

    def test_redirect_uri_refuses_without_an_origin(self) -> None:
        with patch.dict("os.environ", {oidc.PUBLIC_ORIGIN_VAR: ""}), self.assertRaises(OidcError):
            oidc.redirect_uri()


class TestAuthorizationUrl(OidcTestCase):
    def _params(self) -> dict[str, list[str]]:
        url = oidc.authorization_url(
            state="the-state", nonce=NONCE, code_verifier=oidc.new_code_verifier()
        )
        return parse_qs(urlparse(url).query)

    def test_points_at_google(self) -> None:
        url = oidc.authorization_url(state="s", nonce=NONCE, code_verifier="v" * 64)
        self.assertTrue(url.startswith(oidc.AUTH_ENDPOINT))

    def test_carries_everything_the_callback_needs(self) -> None:
        params = self._params()
        self.assertEqual(params["client_id"], [CLIENT_ID])
        self.assertEqual(params["response_type"], ["code"])
        self.assertEqual(params["state"], ["the-state"])
        self.assertEqual(params["nonce"], [NONCE])
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["redirect_uri"], ["https://photos.papalini.casa/auth/callback"])

    def test_requests_only_the_three_exempt_scopes(self) -> None:
        """Any further scope voids Google's verification exemption."""
        self.assertEqual(self._params()["scope"], ["openid email profile"])

    def test_the_verifier_itself_never_leaves(self) -> None:
        verifier = oidc.new_code_verifier()
        url = oidc.authorization_url(state="s", nonce=NONCE, code_verifier=verifier)
        self.assertNotIn(verifier, url)

    def test_challenge_matches_the_published_pkce_example(self) -> None:
        """RFC 7636 appendix B's vector, so the transform is provably right."""
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        self.assertEqual(
            oidc.code_challenge_for(verifier), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        )


class TestValidateClaims(OidcTestCase):
    def _identity(self, **overrides: Any):
        return oidc.validate_claims(_claims(**overrides), expected_nonce=NONCE)

    def test_a_good_token_yields_the_identity(self) -> None:
        identity = self._identity()
        self.assertEqual(identity.google_sub, SUB)
        self.assertEqual(identity.email, EMAIL)
        self.assertEqual(identity.display_name, "Iacopo Papalini")

    def test_bare_issuer_spelling_is_accepted(self) -> None:
        self.assertEqual(self._identity(iss="accounts.google.com").google_sub, SUB)

    def test_foreign_issuer_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(iss="https://evil.example.com")

    def test_token_for_another_client_is_refused(self) -> None:
        """A token minted for a different app must not log anyone in here."""
        with self.assertRaises(OidcError):
            self._identity(aud=OTHER_CLIENT, azp=OTHER_CLIENT)

    def test_audience_as_a_list_is_accepted(self) -> None:
        self.assertEqual(self._identity(aud=[CLIENT_ID]).google_sub, SUB)

    def test_relayed_token_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(azp=OTHER_CLIENT)

    def test_absent_azp_is_fine(self) -> None:
        self.assertEqual(self._identity(azp=_ABSENT).google_sub, SUB)

    def test_expired_token_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(exp=int(time.time()) - 3600)

    def test_token_just_expired_is_tolerated_within_clock_skew(self) -> None:
        self.assertEqual(self._identity(exp=int(time.time()) - 30).google_sub, SUB)

    def test_missing_expiry_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(exp=_ABSENT)

    def test_replayed_token_from_another_login_is_refused(self) -> None:
        """The nonce ties the token to the login that started it."""
        with self.assertRaises(OidcError):
            self._identity(nonce="a-nonce-from-a-different-login")

    def test_missing_nonce_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(nonce=_ABSENT)

    def test_unverified_email_is_refused(self) -> None:
        """The allowlist is keyed on the address, so an unverified one would
        let a stranger claim a family member's invitation."""
        with self.assertRaises(OidcError):
            self._identity(email_verified=False)

    def test_missing_email_verified_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(email_verified=_ABSENT)

    def test_string_true_does_not_pass_for_email_verified(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(email_verified="true")

    def test_missing_subject_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(sub=_ABSENT)

    def test_missing_email_is_refused(self) -> None:
        with self.assertRaises(OidcError):
            self._identity(email=_ABSENT)

    def test_absent_name_is_allowed(self) -> None:
        self.assertIsNone(self._identity(name=_ABSENT).display_name)


class TestDecodeJwtPayload(OidcTestCase):
    def test_reads_a_three_part_token(self) -> None:
        claims = oidc._decode_jwt_payload(_id_token(_claims()))  # noqa: SLF001
        self.assertEqual(claims["sub"], SUB)

    def test_refuses_a_malformed_token(self) -> None:
        for bad in ("", "not-a-jwt", "only.two", "a.b.c.d"):
            with self.subTest(token=bad), self.assertRaises(OidcError):
                oidc._decode_jwt_payload(bad)  # noqa: SLF001

    def test_refuses_a_payload_that_is_not_json(self) -> None:
        token = f"{_b64({'alg': 'RS256'})}.bm90LWpzb24.sig"
        with self.assertRaises(OidcError):
            oidc._decode_jwt_payload(token)  # noqa: SLF001

    def test_refuses_a_payload_that_is_not_an_object(self) -> None:
        payload = base64.urlsafe_b64encode(b'["a","list"]').decode().rstrip("=")
        with self.assertRaises(OidcError):
            oidc._decode_jwt_payload(f"header.{payload}.sig")  # noqa: SLF001
