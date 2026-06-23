# Deploying FaceFind Lite

The app is a single Docker image: FastAPI backend + static frontend + auto-bundled
face models. Pick any free-tier host below.

## 0. Run with Docker locally (or on a $5 VPS)

```bash
docker compose up --build
# → http://localhost:8000
```

Uploads + vectors persist in the `facefind_data` volume. To wipe: `docker compose down -v`.

Or plain Docker:

```bash
docker build -t facefind-lite .
docker run -p 8000:8000 -v facefind_data:/data facefind-lite
```

## 1. Hugging Face Spaces (free)

1. Create a new **Space** → SDK: **Docker**.
2. Push this repo to the Space (the `README.md` frontmatter sets `sdk: docker`, `app_port: 8000`).
3. It builds the Dockerfile and serves the app automatically.

> Free Spaces sleep when idle and have an ephemeral filesystem — uploads reset on
> restart. Fine for demos. The models are baked into the image so there's no
> first-request download.

## 2. Render (free)

1. Push this repo to GitHub.
2. Render dashboard → **New + → Blueprint**, select the repo (`render.yaml` is detected).
3. It builds the Dockerfile, health-checks `/healthz`, and gives you a URL.

> Free tier sleeps after inactivity (cold start) and has no persistent disk.
> Uncomment the `disk:` block in `render.yaml` (paid) for persistence.

## Persistent storage (free, recommended)

The free instance's filesystem is **ephemeral** — by default uploads + face
vectors live in SQLite and reset on every redeploy/sleep. To keep data forever
at **$0/month**, point the app at a free external Postgres:

1. Create a free Postgres on **[Neon](https://neon.tech)** (or Supabase / Render
   Postgres). Copy its connection string, e.g.
   `postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require`.
2. In your host's dashboard, set env var **`DATABASE_URL`** to that string.
3. Redeploy. The app auto-creates its tables on startup and now persists across
   restarts. (No code change — it detects `DATABASE_URL` and switches from SQLite
   to Postgres automatically.)

> Neon's free tier (0.5 GB) never sleeps and holds roughly 1–2k compressed
> photos. For larger libraries, store image bytes in object storage (Cloudflare
> R2 free 10 GB) and keep only vectors in Postgres.

## 3. Fly.io (free allowance)

```bash
fly launch --no-deploy        # detects the Dockerfile
fly volumes create facefind_data --size 1
fly deploy
```

Mount the volume at `/data` (set `FACEFIND_DATA_DIR=/data`, already the default).

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `PORT` | `8000` | Port the server binds to |
| `DATABASE_URL` | _(empty)_ | Postgres connection string for **persistent** storage. If empty, falls back to local SQLite (ephemeral on free hosts). |
| `FACEFIND_DATA_DIR` | `backend/` | Where `facefind.db` (SQLite fallback) lives |
| `FACEFIND_MODELS_DIR` | `backend/models/` | Where ONNX model weights live |
| `FACEFIND_ADMIN_TOKEN` | _(empty)_ | If set, `DELETE /api/reset` and `DELETE /api/photos/{id}` require header `X-Admin-Token`. Leave empty in dev. |
| `FACEFIND_MAX_FILE_MB` | `15` | Max size per uploaded image (returns 413 over limit) |
| `FACEFIND_MAX_FILES` | `30` | Max images per upload request |
| `FACEFIND_ALLOWED_ORIGINS` | `*` | Comma-separated CORS allow-list (set to your domain in prod) |

> **Before public hosting:** set `FACEFIND_ADMIN_TOKEN` to a long random string so
> strangers can't wipe your gallery. In the web UI, paste the same token into the
> "Admin token" field (bottom of the page) to enable delete/reset.

## Cold starts (free tiers)

Free dynos/Spaces sleep when idle; the first request after sleep takes a few seconds.
To remove cold starts, deploy the same image to a small always-on VPS (Tier 2, ~$5/mo)
or ping `/healthz` on a cron.
