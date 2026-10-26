"""Base class for tests that need a database.

``IsolatedAsyncioTestCase`` runs each test in its own event loop, and an
asyncpg pool is bound to the loop that created it — so the pool is built
per test rather than shared. The test's work happens inside a transaction
that is rolled back in teardown, which is what keeps the shared schema
clean without recreating it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase

import asyncpg

from ritrova.db import FaceDB

from .conftest import TEST_DSN

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# A test that opens a second connection while another holds a conflicting lock
# would otherwise wait forever; fail fast so the cause is visible.
_TIMEOUTS = "SET lock_timeout = '5s'; SET statement_timeout = '30s'"


async def _init_connection(connection: asyncpg.Connection) -> None:
    await connection.execute(_TIMEOUTS)


class PostgresTestCase(IsolatedAsyncioTestCase):
    db: FaceDB
    tmp_path: Path

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self._pool = await asyncpg.create_pool(
            TEST_DSN, min_size=1, max_size=1, init=_init_connection
        )
        assert self._pool is not None
        self.db = FaceDB(self._pool, data_dir=self.tmp_path, base_dir=None)
        self._connection = await self._pool.acquire()
        self._transaction = self._connection.transaction()
        await self._transaction.start()
        self._token = self.db._pinned.set(self._connection)  # noqa: SLF001 -- harness owns the pin

    async def asyncTearDown(self) -> None:
        self.db._pinned.reset(self._token)  # noqa: SLF001
        await self._transaction.rollback()
        await self._pool.release(self._connection)
        await self._pool.close()
        self._tmp.cleanup()


# Tables in dependency order; children before parents so TRUNCATE ... CASCADE
# is not needed and the intent stays explicit.
_TABLES = (
    "audit_log",
    "sessions",
    "allowed_users",
    "users",
    "finding_assignment",
    "cluster_findings",
    "print_selection",
    "subject_circles",
    "descriptions",
    "source_path_metadata",
    "findings",
    "scans",
    "sources",
    "subjects",
    "circles",
)


class PostgresAppTestCase(IsolatedAsyncioTestCase):
    """For tests that drive the app or CLI, which open their own pools.

    Rollback isolation cannot work here: a second connection writing to a
    table this test's open transaction holds would block until the test
    times out. These tests commit for real and the tables are truncated
    in teardown instead.
    """

    db: FaceDB
    tmp_path: Path
    dsn: str = TEST_DSN

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self._pool = await asyncpg.create_pool(
            TEST_DSN, min_size=1, max_size=4, init=_init_connection
        )
        assert self._pool is not None
        self.db = FaceDB(self._pool, data_dir=self.tmp_path, base_dir=None)

    def make_client(self, photos_dir: str | None = None) -> TestClient:
        """Build the app and start it, so the lifespan opens its own pool.

        The app runs in TestClient's own thread and loop; it reaches the
        same database as ``self.db``, and both commit, so each sees the
        other's writes.
        """
        from fastapi.testclient import TestClient

        from ritrova.app import create_app

        self.app = create_app(TEST_DSN, self.tmp_path, photos_dir=photos_dir)
        client = TestClient(self.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    async def asyncTearDown(self) -> None:
        await self._pool.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")
        await self._pool.close()
        self._tmp.cleanup()
