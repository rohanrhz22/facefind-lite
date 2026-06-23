"""Data layer for FaceFind Lite.

Supports two backends transparently:
  * SQLite  (default, local dev) — single file `facefind.db`.
  * Postgres (production)        — set DATABASE_URL (e.g. a free Neon/Render DB)
                                   so uploads + face vectors survive restarts.

The rest of the app talks to a thin connection wrapper, so the same helper
functions work against either backend.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = bool(DATABASE_URL)

DATA_DIR = os.environ.get("FACEFIND_DATA_DIR", os.path.dirname(__file__))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "facefind.db")

if USE_PG:
    # psycopg3 accepts postgresql:// — normalize the older postgres:// prefix.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
    import psycopg
    from psycopg.rows import dict_row

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS photo (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT,
    mime        TEXT,
    data        BLOB,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS face (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id    INTEGER REFERENCES photo(id) ON DELETE CASCADE,
    bbox        TEXT,
    score       REAL,
    embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_face_photo ON face(photo_id);
"""

SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS photo (
    id          SERIAL PRIMARY KEY,
    filename    TEXT,
    mime        TEXT,
    data        BYTEA,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS face (
    id          SERIAL PRIMARY KEY,
    photo_id    INTEGER REFERENCES photo(id) ON DELETE CASCADE,
    bbox        TEXT,
    score       REAL,
    embedding   BYTEA
);

CREATE INDEX IF NOT EXISTS idx_face_photo ON face(photo_id);
"""


class _Conn:
    """Uniform connection wrapper over sqlite3 / psycopg with `?` placeholders."""

    def __init__(self, raw, is_pg: bool):
        self.raw = raw
        self.is_pg = is_pg

    def execute(self, sql: str, params=()):
        if self.is_pg:
            cur = self.raw.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            return cur
        return self.raw.execute(sql, params)

    def executescript(self, script: str) -> None:
        if self.is_pg:
            with self.raw.cursor() as cur:
                for stmt in script.split(";"):
                    if stmt.strip():
                        cur.execute(stmt)
        else:
            self.raw.executescript(script)

    def commit(self) -> None:
        self.raw.commit()

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.raw.commit()
        else:
            self.raw.rollback()
        self.raw.close()


def get_db() -> _Conn:
    if USE_PG:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        return _Conn(conn, True)
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return _Conn(conn, False)


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(SCHEMA_PG if USE_PG else SCHEMA_SQLITE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_photo(conn: _Conn, filename: str, mime: str, data: bytes) -> int:
    if conn.is_pg:
        cur = conn.execute(
            "INSERT INTO photo(filename, mime, data, created_at) "
            "VALUES (?,?,?,?) RETURNING id",
            (filename, mime, data, _now()),
        )
        return int(cur.fetchone()["id"])
    cur = conn.execute(
        "INSERT INTO photo(filename, mime, data, created_at) VALUES (?,?,?,?)",
        (filename, mime, data, _now()),
    )
    return int(cur.lastrowid)


def insert_face(conn: _Conn, photo_id: int, bbox: str, score: float,
                embedding: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO face(photo_id, bbox, score, embedding) VALUES (?,?,?,?)",
        (photo_id, bbox, float(score), embedding.astype("float32").tobytes()),
    )


def all_face_embeddings(conn: _Conn):
    """Yield (photo_id, embedding ndarray) for every stored face."""
    for row in conn.execute("SELECT photo_id, embedding FROM face").fetchall():
        yield row["photo_id"], np.frombuffer(
            bytes(row["embedding"]), dtype="float32")


def get_photo(conn: _Conn, photo_id: int):
    return conn.execute(
        "SELECT id, filename, mime, data FROM photo WHERE id = ?", (photo_id,)
    ).fetchone()


def photo_face_counts(conn: _Conn) -> dict[int, int]:
    rows = conn.execute(
        "SELECT photo_id, COUNT(*) c FROM face GROUP BY photo_id"
    ).fetchall()
    return {r["photo_id"]: r["c"] for r in rows}


def list_photos(conn: _Conn):
    return conn.execute(
        "SELECT id, filename, created_at FROM photo ORDER BY id"
    ).fetchall()


def delete_photo(conn: _Conn, photo_id: int) -> None:
    conn.execute("DELETE FROM face WHERE photo_id = ?", (photo_id,))
    conn.execute("DELETE FROM photo WHERE id = ?", (photo_id,))
    conn.commit()


def counts(conn: _Conn) -> tuple[int, int]:
    p = conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"]
    f = conn.execute("SELECT COUNT(*) c FROM face").fetchone()["c"]
    return int(p), int(f)


def reset(conn: _Conn) -> None:
    if conn.is_pg:
        conn.execute("TRUNCATE face, photo RESTART IDENTITY CASCADE")
    else:
        conn.execute("DELETE FROM face")
        conn.execute("DELETE FROM photo")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('face','photo')")
    conn.commit()
