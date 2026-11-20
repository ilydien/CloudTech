# syntax=docker/dockerfile:1
#
# Satu image untuk semua peran: engine (batch), prefect server/worker,
# serving (FastAPI), dashboard (Streamlit), dan nanti Azure Function.
#
# Konsekuensinya image menjadi besar (~2.3 GB). Itu pertukaran yang disengaja:
# satu image = satu versi library = hasil yang reproducible. Kalau nanti Azure
# Function menolak ukuran ini, pecah menjadi dua Dockerfile (engine vs serving)
# TANPA mengubah kode: keduanya hanya butuh pyarrow + pandas + fastapi.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Jakarta

# tzdata: kalender WIB; curl: healthcheck;         ca-certificates: HTTPS
# git: stempel versi kode di run_log (opsional)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tzdata curl ca-certificates git \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# --- torch CPU dulu, di layer terpisah -------------------------------------
# Dipisah supaya rebuild karena perubahan requirements.txt tidak mengunduh
# ulang ~800 MB torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# --- dependency aplikasi ----------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- kode -------------------------------------------------------------------
WORKDIR /app
COPY . /app

# Direktori state (dipakai saat SINK=local) dan home untuk Prefect.
RUN mkdir -p /app/state/data /home/app/.prefect \
    && useradd --uid 1000 --create-home --home-dir /home/app --shell /bin/bash app \
    && chown -R app:app /app /home/app

ENV HOME=/home/app
USER app

# Smoke test: paket bisa diimpor tanpa menyentuh storage/jaringan.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import ihsg; print(ihsg.__version__)" || exit 1

ENTRYPOINT ["python", "-m", "ihsg.cli"]
CMD ["--help"]
