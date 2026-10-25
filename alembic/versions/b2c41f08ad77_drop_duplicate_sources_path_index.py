"""drop the duplicate index on sources.file_path

``sources.file_path`` carries a UNIQUE constraint, and Postgres backs that with
``sources_file_path_key`` — a unique btree on exactly the column that
``idx_sources_path`` indexes. Every lookup, prefix scan and ORDER BY the plain
index could serve is served by the unique one, so it earns nothing and costs a
second write on each insert and update of a path.

The baseline carried it because the SQLite schema did, where it was redundant
for the same reason. Dropped here rather than edited out of the baseline so the
cluster that has already been migrated moves by the same path as a fresh one.

Revision ID: b2c41f08ad77
Revises: a9b07b12530f
Create Date: 2026-10-25

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b2c41f08ad77"
down_revision: str | Sequence[str] | None = "a9b07b12530f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_sources_path")


def downgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(file_path)")
