"""SQLite data layer for FaceFind Lite (matches the docs' zero-cost data model)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

DATA_DIR = os.environ.get(
    "FACEFIND_DATA_DIR", os.path.dirname(__file__))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "facefind.db")

SCHEMA = """
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


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_photo(conn: sqlite3.Connection, filename: str, mime: str,
                 data: bytes) -> int:
    cur = conn.execute(
        "INSERT INTO photo(filename, mime, data, created_at) VALUES (?,?,?,?)",
        (filename, mime, data, _now()),
    )
    return int(cur.lastrowid)


def insert_face(conn: sqlite3.Connection, photo_id: int, bbox: str,
                score: float, embedding: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO face(photo_id, bbox, score, embedding) VALUES (?,?,?,?)",
        (photo_id, bbox, score, embedding.astype("float32").tobytes()),
    )


def all_face_embeddings(conn: sqlite3.Connection):
    """Yield (photo_id, embedding ndarray) for every stored face."""
    for row in conn.execute("SELECT photo_id, embedding FROM face"):
        yield row["photo_id"], np.frombuffer(row["embedding"], dtype="float32")


def get_photo(conn: sqlite3.Connection, photo_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, filename, mime, data FROM photo WHERE id = ?", (photo_id,)
    ).fetchone()


def photo_face_counts(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        "SELECT photo_id, COUNT(*) c FROM face GROUP BY photo_id"
    ).fetchall()
    return {r["photo_id"]: r["c"] for r in rows}


def list_photos(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT id, filename, created_at FROM photo ORDER BY id"
    ).fetchall()


def delete_photo(conn: sqlite3.Connection, photo_id: int) -> None:
    conn.execute("DELETE FROM face WHERE photo_id = ?", (photo_id,))
    conn.execute("DELETE FROM photo WHERE id = ?", (photo_id,))
    conn.commit()


def counts(conn: sqlite3.Connection) -> tuple[int, int]:
    p = conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"]
    f = conn.execute("SELECT COUNT(*) c FROM face").fetchone()["c"]
    return int(p), int(f)


def reset(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM face")
    conn.execute("DELETE FROM photo")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('face','photo')")
    conn.commit()
