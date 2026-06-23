---
title: FaceFind Lite
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# FaceFind Lite

Upload a gallery of photos, then a single selfie — get back **every photo containing that person**.
This is the **Mode B (free-tier server)** build from `FaceFind-Lite-LowCost-Docs.html`,
re-engineered to run at near-zero cost.

## Stack (all free / open-source)

| Layer | Choice | Why |
|-------|--------|-----|
| Face detection | OpenCV **YuNet** | Fast, accurate, tiny ONNX model |
| Face embedding | OpenCV **SFace** | ~99% LFW, 128-d L2-normalized vectors |
| Search | **NumPy** brute-force cosine | No vector DB needed up to ~100k faces |
| Storage / vectors | **SQLite** | Zero-cost, single file |
| API | **FastAPI + Uvicorn** | Async, free-tier friendly |
| Frontend | Static **HTML + JS** | Hosts free anywhere |

> The docs recommend InsightFace/face-api.js. We use OpenCV's YuNet + SFace because it
> installs cleanly from prebuilt wheels (no C++ build step) while keeping the same
> "detect → embed → cosine compare" maths. The face engine in `backend/face_engine.py`
> is isolated, so swapping in InsightFace later is a one-file change.

## Run locally

```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

- The two face models (~37 MB) auto-download from the OpenCV Zoo on first start into `backend/models/`.
- Data is stored in `backend/facefind.db` (SQLite). Delete it (or click **Reset gallery**) to wipe everything.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/photos` | POST | Upload photos → detect + embed + store |
| `/api/search` | POST | Upload 1 selfie → return matching photos |
| `/api/photos/{id}` | GET | Serve a stored image |
| `/api/photos/{id}` | DELETE | Remove one photo + its faces |
| `/api/gallery` | GET | List indexed photos (id, filename, face count) |
| `/api/download` | POST | Bundle selected photo ids into a ZIP |
| `/api/stats` | GET | Gallery + face counts |
| `/healthz` | GET | Liveness probe |
| `/api/reset` | DELETE | Clear gallery + all face data |

### UX features

- **Indexing progress bar** — large galleries upload in batches of 5 with live progress.
- **Match strictness slider** — tune the cosine threshold per search.
- **Download all matches** — one click bundles the matched photos into a ZIP.
- **Full-gallery tab** — browse every indexed photo with face counts; delete any photo inline.
- **Lightbox** — click any result to view full-size and download.

Example search response:

```json
{
  "matches": [
    { "photo_id": 1, "url": "/api/photos/1", "similarity": 0.52 }
  ],
  "total": 1,
  "threshold": 0.363
}
```

`threshold` is SFace's recommended cosine match cutoff (~0.363). Pass `?threshold=` to `/api/search` to tune precision/recall.

## Project layout

```
backend/
  app.py           FastAPI app + endpoints, serves the frontend
  face_engine.py   YuNet + SFace wrapper (pluggable)
  db.py            SQLite data layer (photo + face tables)
  requirements.txt
  models/          auto-downloaded ONNX weights
  facefind.db      SQLite database (created on first run)
frontend/
  index.html       drag-drop gallery + selfie search UI
Dockerfile         container image (bundles models at build)
docker-compose.yml local / VPS deploy with a persistent volume
render.yaml        Render free-tier blueprint
DEPLOY.md          step-by-step hosting guide (Docker, HF Spaces, Render, Fly)
```

## Deploy

One Docker image runs everywhere. Quick start:

```bash
docker compose up --build      # → http://localhost:8000
```

See **[DEPLOY.md](DEPLOY.md)** for Hugging Face Spaces, Render, and Fly.io free-tier steps.

## Privacy

Face data is biometric/sensitive. The **Reset gallery** button performs a one-shot SQL wipe.
For production, add explicit consent, a retention TTL, and per-gallery deletion (see docs §15).
