"""
Test configuration and shared fixtures.

Tests use an in-memory SQLite database so they require no running Postgres.
Qdrant interactions are mocked at the service layer — tests do not require
a running Qdrant instance.

To run: pytest tests/
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock, patch

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass  # fixture handles close

    app.dependency_overrides[get_db] = override_get_db

    # Patch Qdrant and embedding calls so tests are fully offline
    with patch("app.services.vector_store.get_client") as mock_qdrant, \
         patch("app.services.embedder.get_embeddings") as mock_embed, \
         patch("app.services.embedder.embed_query") as mock_query_embed:

        mock_qdrant.return_value = MagicMock()
        mock_embed.return_value = [[0.1] * 1536]
        mock_query_embed.return_value = [0.1] * 1536

        yield TestClient(app)

    app.dependency_overrides.clear()
