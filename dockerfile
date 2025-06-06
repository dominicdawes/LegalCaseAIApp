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
        libreoffice-writer && \
        tesseract-ocr \
        tesseract-ocr-spa \
        tesseract-ocr-fra \
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

# 6) Tell Render how to launch your worker
#    (For example, if you run a Celery worker named `celery_worker.py`)
CMD ["sh", "-c", "celery -A tasks.celery_app worker --loglevel=info --concurrency=2"]
