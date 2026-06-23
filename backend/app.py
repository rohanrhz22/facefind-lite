"""
FaceFind Lite — FastAPI backend (Mode B from the build docs).

Endpoints:
  POST   /api/photos        upload photos -> detect + embed + store
  POST   /api/search        upload 1 selfie -> return matching photos
  GET    /api/photos/{id}   serve a stored image
  GET    /api/gallery       list indexed photos (id, filename, face count)
  POST   /api/download      zip up selected photo ids
  DELETE /api/photos/{id}   remove a single photo + its faces
  GET    /api/stats         gallery + face counts
  GET    /healthz           liveness probe
  DELETE /api/reset         wipe gallery + all face data

Recognition uses brute-force cosine (dot product on L2-normalized vectors)
over all stored faces in NumPy — fast to ~100k faces, no vector DB needed.
"""

from __future__ import annotations

import io
import os
import zipfile

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import asyncio
import db
from face_engine import get_engine

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# ---- Limits & config (override via environment) ----
MAX_FILE_MB = float(os.environ.get("FACEFIND_MAX_FILE_MB", "15"))
MAX_FILES_PER_REQUEST = int(os.environ.get("FACEFIND_MAX_FILES", "30"))
MAX_FILE_BYTES = int(MAX_FILE_MB * 1024 * 1024)
ALLOWED_ORIGINS = [o.strip() for o in
                   os.environ.get("FACEFIND_ALLOWED_ORIGINS", "*").split(",")
                   if o.strip()]
# Protects destructive endpoints (reset, delete). Empty = unprotected (dev).
ADMIN_TOKEN = os.environ.get("FACEFIND_ADMIN_TOKEN", "").strip()

app = FastAPI(title="FaceFind Lite", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_admin(x_admin_token: str | None) -> None:
    """Gate destructive actions behind a token when one is configured."""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token required.")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    # Warm up the model so the first request isn't slow.
    get_engine()


def _read_image(data: bytes) -> np.ndarray:
    """Decode arbitrary image bytes into a BGR ndarray (EXIF-aware)."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        # Respect EXIF orientation.
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        rgb = np.asarray(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")


async def _embed_async(img: np.ndarray):
    """Run blocking CPU inference in a threadpool to keep the loop responsive."""
    return await asyncio.to_thread(get_engine().embed_faces, img)


@app.post("/api/photos")
async def upload_photos(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {MAX_FILES_PER_REQUEST}). "
                   "Upload in smaller batches.")
    results = []
    with db.get_db() as conn:
        for f in files:
            raw = await f.read()
            if not raw:
                continue
            if len(raw) > MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"'{f.filename}' exceeds the {MAX_FILE_MB:g} MB limit.")
            img = _read_image(raw)
            photo_id = db.insert_photo(conn, f.filename or "photo",
                                       f.content_type or "image/jpeg", raw)
            faces = await _embed_async(img)
            for face in faces:
                db.insert_face(conn, photo_id, str(face.bbox), face.score,
                               face.embedding)
            results.append({
                "photo_id": photo_id,
                "filename": f.filename,
                "faces": len(faces),
            })
        conn.commit()
    total_photos, total_faces = _stats()
    return {
        "indexed": results,
        "total_photos": total_photos,
        "total_faces": total_faces,
    }


@app.post("/api/search")
async def search(file: UploadFile = File(...), threshold: float = 0.45):
    """Return every photo containing the person in the uploaded selfie.

    SFace's baseline cosine threshold is ~0.363; we default a bit higher
    (0.45) to cut false positives, and expose it as a tunable parameter.
    """
    engine = get_engine()
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Reference image exceeds the {MAX_FILE_MB:g} MB limit.")
    img = _read_image(raw)
    query_faces = await _embed_async(img)
    if not query_faces:
        return {"matches": [], "total": 0, "threshold": threshold,
                "message": "No face detected in the reference image."}

    # Use the most confident face in the selfie as the query.
    q = max(query_faces, key=lambda fc: fc.score).embedding

    with db.get_db() as conn:
        best: dict[int, float] = {}
        for photo_id, emb in db.all_face_embeddings(conn):
            sim = float(np.dot(q, emb))
            if sim >= threshold and sim > best.get(photo_id, -1.0):
                best[photo_id] = sim

    matches = sorted(
        ({"photo_id": p, "url": f"/api/photos/{p}", "similarity": round(s, 4)}
         for p, s in best.items()),
        key=lambda r: r["similarity"], reverse=True,
    )
    return {"matches": matches, "total": len(matches), "threshold": threshold}


@app.get("/api/photos/{photo_id}")
def get_photo(photo_id: int):
    with db.get_db() as conn:
        row = db.get_photo(conn, photo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    return Response(content=row["data"],
                    media_type=row["mime"] or "image/jpeg")


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: int,
                 x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    with db.get_db() as conn:
        if db.get_photo(conn, photo_id) is None:
            raise HTTPException(status_code=404, detail="Photo not found")
        db.delete_photo(conn, photo_id)
    return {"status": "ok", "deleted": photo_id}


@app.get("/api/gallery")
def gallery():
    with db.get_db() as conn:
        photos = db.list_photos(conn)
        face_counts = db.photo_face_counts(conn)
    return {
        "photos": [
            {"photo_id": p["id"], "filename": p["filename"],
             "url": f"/api/photos/{p['id']}",
             "faces": face_counts.get(p["id"], 0)}
            for p in photos
        ]
    }


@app.post("/api/download")
def download(photo_ids: list[int] = Body(..., embed=True)):
    """Bundle the given photo ids into a single ZIP download."""
    if not photo_ids:
        raise HTTPException(status_code=400, detail="No photo ids provided.")
    buf = io.BytesIO()
    seen_names: set[str] = set()
    with db.get_db() as conn, zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for pid in photo_ids:
            row = db.get_photo(conn, pid)
            if row is None:
                continue
            name = row["filename"] or f"photo_{pid}.jpg"
            # Avoid clashes when filenames repeat.
            if name in seen_names:
                base, ext = os.path.splitext(name)
                name = f"{base}_{pid}{ext}"
            seen_names.add(name)
            zf.writestr(name, row["data"])
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="facefind-matches.zip"'},
    )


@app.get("/api/stats")
def stats():
    photos, faces = _stats()
    return {"total_photos": photos, "total_faces": faces}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.delete("/api/reset")
def reset(x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    with db.get_db() as conn:
        db.reset(conn)
    return {"status": "ok", "message": "Gallery and all face data cleared."}


def _stats() -> tuple[int, int]:
    with db.get_db() as conn:
        return db.counts(conn)


# Serve the frontend (index.html) at the root.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
