"""Adaptor HTTP lokal (pengganti Azure Function saat pengembangan).

Handler yang sama dipakai oleh Azure Function (``function_app/``), jadi perilaku
lokal dan produksi tidak bisa menyimpang.
"""

from __future__ import annotations

import os
import subprocess
import sys

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from ihsg import __version__
from ihsg.config import Settings
from ihsg.serving import handler
from ihsg.storage import get_sink

app = FastAPI(
    title="IHSG Forecasting Engine",
    version=__version__,
    description="Ramalan volatilitas & return IHSG 1..7 hari (data gold, bukan inference ulang).",
)

settings = Settings.from_env()


@app.get("/health")
def health() -> dict:
    return handler.health(settings)


@app.get("/forecast/latest")
def forecast_latest() -> JSONResponse:
    status, body = handler.handle_request(settings, None)
    return JSONResponse(status_code=status, content=body)


@app.get("/forecast")
def forecast(as_of: str | None = Query(default=None, description="YYYY-MM-DD")) -> JSONResponse:
    status, body = handler.handle_request(settings, as_of)
    return JSONResponse(status_code=status, content=body)


@app.get("/runs")
def runs(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    from ihsg.runlog import read_runs

    table = read_runs(get_sink(settings))
    if table.empty:
        return {"runs": []}
    table = table.sort_values("as_of", ascending=False)
    columns = ["as_of", "stage", "run_id", "trigger", "promoted", "val_metric", "duration_s"]
    return {"runs": table[columns].head(limit).to_dict(orient="records")}


@app.post("/admin/retrain", status_code=202)
def retrain(x_admin_token: str | None = Header(default=None)) -> dict:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN belum di-set")
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="token tidak valid")

    # Detached: request tidak menunggu training selesai.
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "ihsg.cli", "run", "--retrain", "--trigger", "api"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    return {"status": "accepted", "pid": process.pid, "trigger": "api"}
