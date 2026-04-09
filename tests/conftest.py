import pytest

from face_recog.db import FaceDB


@pytest.fixture
def db(tmp_path):
    db_instance = FaceDB(tmp_path / "test.db")
    yield db_instance
    db_instance.close()
