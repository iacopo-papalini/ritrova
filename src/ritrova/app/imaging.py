"""Bounded thread pool for blocking image work (ADR-012 D3).

Decoding a JPEG, cropping a face, or seeking a video frame holds the GIL
for long enough to stall every other request on the single event loop.
These calls run on a small, fixed pool instead: the pool size caps how
many decodes happen at once, and callers simply await their turn, so a
burst of thumbnail requests queues rather than exhausting memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_executor: ThreadPoolExecutor | None = None


def configure(max_workers: int) -> None:
    global _executor
    _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ritrova-img")


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


async def run(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking imaging call on the pool and await the result."""
    if _executor is None:
        raise RuntimeError("imaging.configure(...) has not been called — use create_app()")
    if kwargs:
        msg = "imaging.run takes positional arguments only"
        raise TypeError(msg)
    return await asyncio.get_running_loop().run_in_executor(_executor, fn, *args)
