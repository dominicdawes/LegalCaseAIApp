# ────────────────────────────────────────────────────────────────────────────
# Dockerfile
# ────────────────────────────────────────────────────────────────────────────

# 1) Pick a base image that has Debian/Ubuntu under the hood
FROM python:3.11-slim

# 2) Install LibreOffice (and any other system dependencies) in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        antiword \
        catdoc \
        libreoffice-core \
        libreoffice-common \
        libreoffice-writer \
        tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3) Create and activate a virtual environment (optional, but recommended)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4) Copy your requirements.txt and install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Downgrade pip to <24.1 so that it doesn’t reject textract’s metadata
RUN pip install pip==24.0 \
    && pip install -r requirements.txt

# 5) Copy in your application code
COPY . /app

# 6) Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV CELERY_HIJACK_ROOT_LOGGER=0
ENV CELERY_LOG_LEVEL=INFO

# 7) Tell Render how to launch your worker with explicit logging (Purge queues on startup && Start worker + queues)
#    (For example, if you run a Celery worker named `celery_worker.py`)
#    CMD ["sh", "-c", "celery -A tasks.celery_app worker --loglevel=info --concurrency=2"]
CMD ["sh", "-c", "echo '🧹 Purging queues on startup...' && celery -A tasks.celery_app purge -f && echo '✅ Queues purged, starting worker...' && celery -A tasks.celery_app worker --loglevel=info --concurrency=100 -Q celery,ingest,parsing,embedding,finalize -P gevent"]
# CMD ["sh", "-c", "celery -A tasks.celery_app worker --loglevel=info -P gevent --concurrency=2 -Q celery,ingest,parsing,embedding,finalize --without-gossip --without-mingle --without-heartbeat"]