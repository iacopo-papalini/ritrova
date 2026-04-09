from collections.abc import Generator
from pathlib import Path

import pytest

from face_recog.db import FaceDB


@pytest.fixture
def db(tmp_path: Path) -> Generator[FaceDB]:
    db_instance = FaceDB(tmp_path / "test.db")
    yield db_instance
    db_instance.close()
