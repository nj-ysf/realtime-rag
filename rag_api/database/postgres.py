import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from rag_api.config import Settings
from rag_api.core.exceptions import DatabaseError

logger = logging.getLogger("rag_api.database")


class PostgresDatabase:
    """
    Opens one short-lived, read-only connection per unit of work.

    The API only reads, so the session is read-only; connections are always closed,
    even on errors. Error messages never include connection details or credentials.
    """

    def __init__(self, settings: Settings) -> None:
        self._connect_kwargs = {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "dbname": settings.postgres_db,
            "user": settings.postgres_user,
            "password": settings.postgres_password or None,
            "connect_timeout": settings.postgres_connect_timeout,
        }

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        try:
            conn = psycopg.connect(**self._connect_kwargs, row_factory=dict_row, autocommit=True)
        except psycopg.Error as exc:
            logger.error("PostgreSQL connection failed: %s", type(exc).__name__)
            raise DatabaseError("Could not connect to PostgreSQL") from exc

        try:
            conn.read_only = True
            yield conn
        finally:
            conn.close()

    def ping(self) -> None:
        try:
            with self.connection() as conn:
                conn.execute("SELECT 1")
        except psycopg.Error as exc:
            raise DatabaseError("PostgreSQL health check failed") from exc
