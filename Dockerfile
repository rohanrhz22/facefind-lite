# FaceFind Lite — Mode B (free-tier server) container image.
FROM python:3.12-slim

# OpenCV runtime needs these shared libs even with the headless-friendly wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFIND_DATA_DIR=/data \
    FACEFIND_MODELS_DIR=/app/backend/models \
    PORT=8000

WORKDIR /app

# Install deps first for better layer caching.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Pre-download the YuNet + SFace models at build time so the first request
# isn't blocked on a ~37 MB download (avoids free-tier cold-start timeouts).
RUN python -c "import sys; sys.path.insert(0,'backend'); from face_engine import get_engine; get_engine()"

# Persist uploads/vectors on a mounted volume.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Shell form so $PORT (set by Render/Fly/HF) is expanded at runtime.
CMD ["sh", "-c", "cd backend && uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
