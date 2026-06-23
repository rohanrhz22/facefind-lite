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

CREATE TABLE IF NOT EXISTS person (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    token       TEXT UNIQUE,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detection_photo (
    detection_id INTEGER REFERENCES detection(id) ON DELETE CASCADE,
    photo_id     INTEGER REFERENCES photo(id) ON DELETE CASCADE,
    similarity   REAL
);

CREATE INDEX IF NOT EXISTS idx_detphoto_det ON detection_photo(detection_id);
CREATE INDEX IF NOT EXISTS idx_detphoto_photo ON detection_photo(photo_id);
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

CREATE TABLE IF NOT EXISTS person (
    id          SERIAL PRIMARY KEY,
    name        TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detection (
    id          SERIAL PRIMARY KEY,
    name        TEXT,
    token       TEXT UNIQUE,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS detection_photo (
    detection_id INTEGER REFERENCES detection(id) ON DELETE CASCADE,
    photo_id     INTEGER REFERENCES photo(id) ON DELETE CASCADE,
    similarity   REAL
);

CREATE INDEX IF NOT EXISTS idx_detphoto_det ON detection_photo(detection_id);
CREATE INDEX IF NOT EXISTS idx_detphoto_photo ON detection_photo(photo_id);
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
        # Migration: add face.person_id (nullable FK) if it doesn't exist yet.
        try:
            conn.execute(
                "ALTER TABLE face ADD COLUMN person_id INTEGER "
                "REFERENCES person(id) ON DELETE SET NULL")
            conn.commit()
        except Exception:
            # Column already exists (SQLite raises; PG raises duplicate_column).
            conn.raw.rollback() if conn.is_pg else None


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
        conn.execute(
            "TRUNCATE detection_photo, detection, face, photo, person "
            "RESTART IDENTITY CASCADE")
    else:
        conn.execute("DELETE FROM detection_photo")
        conn.execute("DELETE FROM detection")
        conn.execute("DELETE FROM face")
        conn.execute("DELETE FROM photo")
        conn.execute("DELETE FROM person")
        conn.execute(
            "DELETE FROM sqlite_sequence "
            "WHERE name IN ('face','photo','person','detection')")
    conn.commit()


# ---- Detections (user "find my photos" records) ----

def create_detection(conn: _Conn, name: str, token: str) -> int:
    if conn.is_pg:
        cur = conn.execute(
            "INSERT INTO detection(name, token, created_at) "
            "VALUES (?,?,?) RETURNING id", (name, token, _now()))
        return int(cur.fetchone()["id"])
    cur = conn.execute(
        "INSERT INTO detection(name, token, created_at) VALUES (?,?,?)",
        (name, token, _now()))
    return int(cur.lastrowid)


def add_detection_photo(conn: _Conn, detection_id: int, photo_id: int,
                        similarity: float) -> None:
    conn.execute(
        "INSERT INTO detection_photo(detection_id, photo_id, similarity) "
        "VALUES (?,?,?)", (detection_id, photo_id, float(similarity)))


def detection_id_for_token(conn: _Conn, token: str):
    row = conn.execute(
        "SELECT id FROM detection WHERE token = ?", (token,)).fetchone()
    return row["id"] if row else None


def photo_in_detection(conn: _Conn, token: str, photo_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM detection d JOIN detection_photo dp "
        "ON dp.detection_id = d.id "
        "WHERE d.token = ? AND dp.photo_id = ? LIMIT 1",
        (token, photo_id)).fetchone()
    return row is not None


def detection_photo_ids(conn: _Conn, token: str):
    rows = conn.execute(
        "SELECT dp.photo_id FROM detection d JOIN detection_photo dp "
        "ON dp.detection_id = d.id WHERE d.token = ? ORDER BY dp.photo_id",
        (token,)).fetchall()
    return [r["photo_id"] for r in rows]


def list_detections(conn: _Conn):
    """Admin view: every user who found themselves, with photo counts."""
    rows = conn.execute("""
        SELECT d.id, d.name, d.token, d.created_at,
               COUNT(dp.photo_id) AS photo_count
        FROM detection d
        LEFT JOIN detection_photo dp ON dp.detection_id = d.id
        GROUP BY d.id, d.name, d.token, d.created_at
        ORDER BY d.created_at DESC, d.id DESC
    """).fetchall()
    out = []
    for r in rows:
        pids = [pr["photo_id"] for pr in conn.execute(
            "SELECT photo_id FROM detection_photo "
            "WHERE detection_id = ? ORDER BY photo_id", (r["id"],)).fetchall()]
        out.append({
            "id": r["id"], "name": r["name"], "token": r["token"],
            "created_at": r["created_at"], "photo_count": r["photo_count"],
            "photo_ids": pids,
        })
    return out


# ---- People / clustering helpers ----

def all_faces(conn: _Conn):
    """Return list of (face_id, photo_id, embedding ndarray) for every face."""
    out = []
    for row in conn.execute(
            "SELECT id, photo_id, embedding FROM face ORDER BY id").fetchall():
        out.append((row["id"], row["photo_id"],
                    np.frombuffer(bytes(row["embedding"]), dtype="float32")))
    return out


def create_person(conn: _Conn, name: str | None) -> int:
    if conn.is_pg:
        cur = conn.execute(
            "INSERT INTO person(name, created_at) VALUES (?,?) RETURNING id",
            (name, _now()))
        return int(cur.fetchone()["id"])
    cur = conn.execute(
        "INSERT INTO person(name, created_at) VALUES (?,?)", (name, _now()))
    return int(cur.lastrowid)


def rename_person(conn: _Conn, person_id: int, name: str) -> None:
    conn.execute("UPDATE person SET name = ? WHERE id = ?", (name, person_id))
    conn.commit()


def assign_face_person(conn: _Conn, face_id: int, person_id: int) -> None:
    conn.execute("UPDATE face SET person_id = ? WHERE id = ?",
                 (person_id, face_id))


def clear_people(conn: _Conn) -> None:
    """Detach all faces from people and remove all person rows."""
    conn.execute("UPDATE face SET person_id = NULL")
    conn.execute("DELETE FROM person")
    if not conn.is_pg:
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'person'")
    conn.commit()


def list_people(conn: _Conn):
    """People with their face/photo counts and a representative photo."""
    rows = conn.execute("""
        SELECT p.id, p.name,
               COUNT(f.id)                AS face_count,
               COUNT(DISTINCT f.photo_id) AS photo_count,
               MIN(f.photo_id)            AS rep_photo_id
        FROM person p
        LEFT JOIN face f ON f.person_id = p.id
        GROUP BY p.id, p.name
        HAVING COUNT(f.id) > 0
        ORDER BY face_count DESC, p.id
    """).fetchall()
    return rows


def person_photo_ids(conn: _Conn, person_id: int):
    rows = conn.execute(
        "SELECT DISTINCT photo_id FROM face WHERE person_id = ? "
        "ORDER BY photo_id", (person_id,)).fetchall()
    return [r["photo_id"] for r in rows]
