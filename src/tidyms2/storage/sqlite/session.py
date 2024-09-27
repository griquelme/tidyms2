"""SQLAlchemy session management utilities."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session, sessionmaker


@contextmanager
def create_session(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Get a SQLALchemy session and close connection after usage."""
    with factory() as session:
        transaction = session.begin()
        try:
            yield session
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            raise e
