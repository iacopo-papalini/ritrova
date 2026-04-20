"""301 redirects for the pre-``/{kind}/...`` URL scheme.

Kept so pre-refactor bookmarks keep working. Paths like ``/clusters`` are
currently shadowed by the ``/{kind}`` catch-all in ``pages`` (which 422s
on unknown kinds), matching historical behaviour — but these handlers
stay registered so that when ``pages`` stops catching a bare top-level
name (e.g. after tightening the kind type), the redirects kick in.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from ..deps import get_db
from ..helpers import kind_for_subject

router = APIRouter()


@router.get("/clusters", response_class=RedirectResponse)
def redirect_clusters(species: str = "human") -> RedirectResponse:
    kind = "pets" if species == "pet" else "people"
    return RedirectResponse(f"/{kind}/clusters", status_code=301)


@router.get("/persons", response_class=RedirectResponse)
def redirect_persons(species: str = "human") -> RedirectResponse:
    kind = "pets" if species == "pet" else "people"
    return RedirectResponse(f"/{kind}", status_code=301)


@router.get("/persons/{person_id}", response_class=RedirectResponse)
def redirect_person_detail(person_id: int) -> RedirectResponse:
    subject = get_db().get_subject(person_id)
    kind = kind_for_subject(subject.kind) if subject else "people"
    return RedirectResponse(f"/{kind}/{person_id}", status_code=301)


@router.get("/persons/{person_id}/find-similar", response_class=RedirectResponse)
def redirect_find_similar(person_id: int) -> RedirectResponse:
    subject = get_db().get_subject(person_id)
    kind = kind_for_subject(subject.kind) if subject else "people"
    return RedirectResponse(f"/{kind}/{person_id}/find-similar", status_code=301)


@router.get("/singletons", response_class=RedirectResponse)
def redirect_singletons(species: str = "human") -> RedirectResponse:
    kind = "pets" if species == "pet" else "people"
    return RedirectResponse(f"/{kind}/singletons", status_code=301)


@router.get("/merge-suggestions", response_class=RedirectResponse)
def redirect_merge_suggestions(species: str = "human") -> RedirectResponse:
    kind = "pets" if species == "pet" else "people"
    return RedirectResponse(f"/{kind}/merge-suggestions", status_code=301)


@router.get("/compare", response_class=RedirectResponse)
def redirect_compare(species: str = "human") -> RedirectResponse:
    kind = "pets" if species == "pet" else "people"
    return RedirectResponse(f"/{kind}/compare", status_code=301)
