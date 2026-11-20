# =============================================================================
# IHSG Forecasting Engine - perintah harian
#
# Semua perintah dijalankan lewat Docker Compose; tidak ada yang butuh venv
# lokal. `make` tanpa argumen menampilkan daftar target.
# =============================================================================
COMPOSE := docker compose
ENGINE  := $(COMPOSE) --profile run run --rm engine
AS_OF   ?= auto
START   ?=
END     ?=
COMPOSE_RUN := $(COMPOSE) --profile run run --rm engine

.DEFAULT_GOAL := help
.PHONY: help up down ps logs doctor ingest silver features train predict run backfill retrain4 serve mlflow dash test lint fmt clean shell diagram

help:  ## Tampilkan daftar target
	@grep -E '^[a-zA-Z0-9_]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

up:  ## Nyalakan storage + Prefect + API (bukan batch job)
	$(COMPOSE) up -d
	@echo "Garage console : http://localhost:3909"
	@echo "Prefect        : http://localhost:4200"
	@echo "API            : http://localhost:8080/health"

ps:  ## Status container
	$(COMPOSE) ps

logs:  ## Ikuti log semua service
	$(COMPOSE) logs -f --tail=100

down:  ## Hentikan semua service (volume tetap)
	$(COMPOSE) --profile ui --profile run down

doctor:  ## Cek konektivitas storage & konfigurasi
	$(COMPOSE_RUN) doctor

ingest:  ## Unduh bronze (append-only)
	$(COMPOSE_RUN) ingest --as-of $(AS_OF)

silver:  ## Bangun silver dari bronze
	$(COMPOSE_RUN) silver --as-of $(AS_OF)

features:  ## Bangun gold/features + params
	$(COMPOSE_RUN) features --as-of $(AS_OF)

train:  ## Latih GRU, evaluasi, gate promosi
	$(COMPOSE_RUN) train --as-of $(AS_OF) --trigger manual

predict:  ## Ramalan 1..7 hari (butuh champion)
	$(COMPOSE_RUN) predict --as-of $(AS_OF)

run:  ## Pipeline lengkap sekali jalan (ingest -> predict)
	$(COMPOSE_RUN) run --as-of $(AS_OF)

backfill:  ## Backfill rentang tanggal: make backfill START=2026-10-01 END=2026-10-31
	@test -n "$(START)" || (echo "START wajib diisi, contoh: make backfill START=2026-10-01 END=2026-10-31"; exit 1)
	$(COMPOSE_RUN) backfill --start $(START) --end $(END) --retrain

retrain4:  ## Bukti bonus: 4 retrain pada 4 tanggal berbeda (idempoten per tanggal)
	@for d in 2026-10-09 2026-10-16 2026-10-23 2026-10-30; do \
		echo "== retrain $$d =="; \
		$(COMPOSE_RUN) run --as-of $$d --retrain --trigger manual || exit 1; \
	done
	$(COMPOSE_RUN) report

serve:  ## Nyalakan API lokal
	$(COMPOSE) up -d serving

mlflow:  ## Buka MLflow UI (http://localhost:5000)
	$(COMPOSE) up -d mlflow
	@echo "MLflow: http://localhost:5000"

dash:  ## Nyalakan dashboard Streamlit
	$(COMPOSE) --profile ui up -d dashboard
	@echo "Dashboard: http://localhost:8501"

test:  ## Jalankan unit test di dalam container
	$(COMPOSE) run --rm --no-deps --entrypoint python engine -m pytest -q

lint:  ## Ruff check
	$(COMPOSE) run --rm --no-deps --entrypoint ruff engine check .

fmt:  ## Ruff format
	$(COMPOSE) run --rm --no-deps --entrypoint ruff engine format .

shell:  ## Shell interaktif di image engine
	$(COMPOSE) run --rm --no-deps --entrypoint bash engine

diagram:  ## Render docs/*.dot menjadi PNG + SVG (butuh Docker, tanpa install lokal)
	docker run --rm -v "$(CURDIR)/docs:/w" -w /w python:3.11-slim sh -c \
		"apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq graphviz fonts-dejavu-core >/dev/null 2>&1 && \
		 dot -Tpng -Gdpi=110 architecture.dot -o architecture.png && \
		 dot -Tsvg architecture.dot -o architecture.svg && \
		 dot -Tpng -Gdpi=110 dataflow.dot -o dataflow.png && \
		 dot -Tsvg dataflow.dot -o dataflow.svg"
	@echo "selesai: docs/architecture.{png,svg}, docs/dataflow.{png,svg}"

clean:  ## Hapus container + volume (data hilang!)
	$(COMPOSE) --profile ui --profile run down -v
