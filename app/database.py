import uuid as _uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import String, TypeDecorator

from app.config import settings


class GUID(TypeDecorator):
    """UUID type that works with both PostgreSQL and SQLite.

    PostgreSQL: stored as native UUID via VARCHAR(36).
    SQLite: stored as VARCHAR(36) string.
    Accepts uuid.UUID objects or UUID strings as input.
    Always returns uuid.UUID objects.
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, _uuid.UUID):
            return str(value)
        return str(_uuid.UUID(str(value)))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return _uuid.UUID(str(value))

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
if _is_sqlite:
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

# expire_on_commit=False: ORM objects remain accessible after db.commit()
# without issuing additional SELECT statements. Required for background tasks
# where the session may be used after a commit and before a refresh.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db():
    # Do NOT pass this session into BackgroundTasks. Open a fresh
    # SessionLocal() inside the task instead (see tasks/ingest_task.py).
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
