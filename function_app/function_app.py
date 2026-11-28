"""Azure Function (HTTP trigger) - pembungkus tipis handler yang sama.

Fungsi ini TIDAK menghitung model apa pun: ia membaca tabel gold (Parquet) dan
mengembalikan JSON. Semua komputasi ML terjadi di Azure Container Apps Job.

Bundling & deploy (ringkas):
    make func-bundle
    cd function_app && func azure functionapp publish fn-ihsg-kel3

App Settings yang wajib di Azure:
    SINK=azure
    AZURE_STORAGE_ACCOUNT_URL=https://<storage>.blob.core.windows.net
    AzureWebJobsStorage__accountName=<storage>
    AzureWebJobsStorage__credential=managedidentity
    IDX_HOLIDAYS=2026-01-01,...
Opsional untuk POST /api/admin/retrain:
    ADMIN_TOKEN=<rahasia>
    ACA_JOB_RESOURCE_ID=/subscriptions/.../providers/Microsoft.App/jobs/job-ihsg-weekly
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

import azure.functions as func

from ihsg.config import Settings
from ihsg.runlog import read_runs
from ihsg.serving import handler
from ihsg.storage import get_sink

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


def _json_response(body: object, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False, allow_nan=False, default=str),
        status_code=status_code,
        mimetype="application/json",
    )


def _clean_records(records: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for row in records:
        cleaned.append(
            {
                key: (None if isinstance(val, float) and not math.isfinite(val) else val)
                for key, val in row.items()
            }
        )
    return cleaned


def _trigger_aca_job(job_resource_id: str, api_version: str = "2024-03-01") -> dict:
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default").token
    url = f"https://management.azure.com{job_resource_id}?api-version={api_version}"
    if "/start" not in job_resource_id:
        url = f"https://management.azure.com{job_resource_id.rstrip('/')}/start?api-version={api_version}"

    request = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read().decode("utf-8").strip()
        return json.loads(raw) if raw else {"status": "started"}


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    settings = Settings.from_env()
    return _json_response(handler.health(settings), status_code=200)


@app.route(route="forecast/latest", methods=["GET"])
def forecast_latest(req: func.HttpRequest) -> func.HttpResponse:
    settings = Settings.from_env()
    status, body = handler.handle_request(settings, None)
    return _json_response(body, status_code=status)


@app.route(route="forecast", methods=["GET"])
def forecast(req: func.HttpRequest) -> func.HttpResponse:
    settings = Settings.from_env()
    as_of = req.params.get("as_of")
    status, body = handler.handle_request(settings, as_of)
    return _json_response(body, status_code=status)


@app.route(route="runs", methods=["GET"])
def runs(req: func.HttpRequest) -> func.HttpResponse:
    settings = Settings.from_env()
    try:
        limit = int(req.params.get("limit") or 20)
    except ValueError:
        return _json_response({"status": "error", "error": "limit harus bilangan bulat"}, status_code=400)
    limit = max(1, min(limit, 200))

    table = read_runs(get_sink(settings))
    if table.empty:
        return _json_response({"runs": []}, status_code=200)

    table = table.sort_values("as_of", ascending=False)
    columns = ["as_of", "stage", "run_id", "trigger", "promoted", "val_metric", "duration_s"]
    records = _clean_records(table[columns].head(limit).to_dict(orient="records"))
    return _json_response({"runs": records}, status_code=200)


@app.route(route="admin/retrain", methods=["POST"])
def retrain(req: func.HttpRequest) -> func.HttpResponse:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        return _json_response({"status": "unavailable", "error": "ADMIN_TOKEN belum di-set"}, status_code=503)

    provided = req.headers.get("x-admin-token") or req.headers.get("X-Admin-Token")
    if provided != expected:
        return _json_response({"status": "unauthorized", "error": "token tidak valid"}, status_code=401)

    job_id = (os.getenv("ACA_JOB_RESOURCE_ID") or "").strip()
    if not job_id:
        return _json_response(
            {"status": "unavailable", "error": "ACA_JOB_RESOURCE_ID belum di-set"},
            status_code=503,
        )

    api_version = (os.getenv("ACA_API_VERSION") or "2024-03-01").strip()
    try:
        result = _trigger_aca_job(job_id, api_version=api_version)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return _json_response(
            {"status": "error", "error": f"ACA Job HTTP {error.code}: {detail}"},
            status_code=502,
        )
    except Exception as error:  # noqa: BLE001
        return _json_response(
            {"status": "error", "error": f"{type(error).__name__}: {error}"},
            status_code=502,
        )

    return _json_response({"status": "accepted", "trigger": "api", "job": result}, status_code=202)
