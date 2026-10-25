"""Postgres-backed test fixtures.

The schema is built once per session; each test then runs inside a
transaction that is rolled back afterwards, so the suite shares one
database without a per-test CREATE DATABASE. That works because
``FaceDB.transaction()`` joins an already-open transaction on the pinned
connection rather than starting a second one — the fixture opens the
outermost one and nothing inside can commit past it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg
import pytest

from ritrova.db import FaceDB

ADMIN_DSN = os.environ.get(
    "RITROVA_TEST_ADMIN_DSN", "postgresql://ritrova:ritrova-dev@127.0.0.1:5432/ritrova"
)
TEST_DB = os.environ.get("RITROVA_TEST_DB", "ritrova_test")
TEST_DSN = ADMIN_DSN.rsplit("/", 1)[0] + "/" + TEST_DB

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def _build_schema() -> None:
    from alembic.config import Config

    from alembic import command

    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await admin.close()

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    os.environ["RITROVA_DSN"] = TEST_DSN
    # Alembic's env.py drives its own asyncio.run, which cannot nest.
    await asyncio.to_thread(command.upgrade, config, "head")

    # Committed outside any test's transaction, so every test sees it.
    pool = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=1)
    assert pool is not None
    await FaceDB(pool, data_dir=PROJECT_ROOT, base_dir=None).sync_scan_types()
    await pool.close()


async def _drop_schema() -> None:
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
    finally:
        await admin.close()


@pytest.fixture
async def db(tmp_path: Path):
    """Function-style tests get a committing database, truncated afterwards.

    Class-based tests use ``PostgresTestCase`` instead, which is faster
    because it rolls back rather than truncating.
    """
    from ._postgres import _TABLES, _init_connection

    pool = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=4, init=_init_connection)
    assert pool is not None
    try:
        yield FaceDB(pool, data_dir=tmp_path, base_dir=None)
    finally:
        await pool.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")
        await pool.close()


@pytest.fixture
def dsn() -> str:
    return TEST_DSN


@pytest.fixture(scope="session", autouse=True)
def _schema() -> object:
    """Sync on purpose: no loop is running here, so each test case is free to
    own its own (``IsolatedAsyncioTestCase`` makes a fresh one per test)."""
    asyncio.run(_build_schema())
    yield None
    asyncio.run(_drop_schema())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the Playwright tests last.

    Playwright's sync API keeps an event loop running in the main thread for
    as long as it is loaded, and ``IsolatedAsyncioTestCase`` then refuses to
    start its own ("Runner.run() cannot be called from a running event loop").
    Sorting the browser tests to the end keeps one `pytest` run working.
    """
    items.sort(key=lambda item: "e2e" in item.nodeid.split("::")[0])
