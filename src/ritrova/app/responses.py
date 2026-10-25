"""JSON response that knows how to render the types the DB layer returns.

Starlette's ``JSONResponse`` uses ``json.dumps``, which raises on ``UUID``
and ``datetime`` — both of which every id- or date-carrying endpoint now
returns (D13, D14). Routing through FastAPI's ``jsonable_encoder`` keeps
UUIDs as plain strings and datetimes as ISO-8601.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse as _StarletteJSONResponse


class JSONResponse(_StarletteJSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            jsonable_encoder(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
