"""Azure Function (HTTP trigger) - pembungkus tipis handler yang sama.

Fungsi ini TIDAK menghitung apa pun: ia membaca tabel gold (Parquet) dan
mengembalikan JSON. Semua komputasi terjadi di job ``predict``.

Deploy (ringkas):
    az functionapp create --consumption-plan-location southeastasia \
      --runtime python --runtime-version 3.11 --functions-version 4 \
      --name ihsg-fn --storage-account <sa>
    func azure functionapp publish ihsg-fn

Environment yang wajib di App Settings: SINK, S3_ENDPOINT_URL (atau endpoint
Azure Blob), S3_ACCESS_KEY, S3_SECRET_KEY, IDX_HOLIDAYS.
"""

from __future__ import annotations

import json

import azure.functions as func

from ihsg.config import Settings
from ihsg.serving import handler

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
settings = Settings.from_env()


@app.route(route="forecast", methods=["GET"])
def forecast(req: func.HttpRequest) -> func.HttpResponse:
    as_of = req.params.get("as_of")
    status, body = handler.handle_request(settings, as_of)
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False),
        status_code=status,
        mimetype="application/json",
    )


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(handler.health(settings), ensure_ascii=False),
        status_code=200,
        mimetype="application/json",
    )
