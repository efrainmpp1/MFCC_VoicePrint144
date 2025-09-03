# ===== Base Python =====
FROM python:3.12-slim AS base

# Avoid prompts and improve logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ===== System dependencies =====
# - libsndfile1: required for soundfile/librosa
# - ffmpeg (optional, useful for audio scenarios — can remove if not used)
# - build-essential + libffi-dev: for compiling native wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== Working directory =====
WORKDIR /app

# ===== Copy requirements and install =====
# Copy only requirements first to leverage cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ===== Copy code =====
COPY . /app

# ===== Default environment variables (can be overridden at runtime) =====
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=8000 \
    DEFAULT_MODE=mfcc \
    DEFAULT_PCEN=0 \
    DEFAULT_DOWN16K=1 \
    UPLOAD_DIR=api/uploads

# Ensure upload folder exists
RUN mkdir -p ${UPLOAD_DIR}

# ===== Expose API port =====
EXPOSE 8000

# ===== Simple healthcheck =====
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  wget -qO- http://127.0.0.1:${FLASK_PORT}/health || exit 1

# ===== Execution command (gunicorn) =====
# 2 sync workers are enough to start; adjust according to load/CPU.
# Tip: in production, use --workers (2*CPU)+1 as a baseline.
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "api.wsgi:app"]
