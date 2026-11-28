from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

from ihsg import dataset
from ihsg.runlog import append_run
from ihsg.schemas import GOLD_METRICS, GOLD_PARAMS, GOLD_PREDICTIONS, GOLD_PUBLISHED
from ihsg.storage import LocalSink


def _install_fake_azure_functions(monkeypatch):
    class FakeHttpRequest:
        def __init__(self, *, params=None, headers=None):
            self.params = params or {}
            self.headers = headers or {}

    class FakeHttpResponse:
        def __init__(self, body, status_code=200, mimetype="application/json"):
            self.body = body
            self.status_code = status_code
            self.mimetype = mimetype

    class FakeFunctionApp:
        def __init__(self, http_auth_level=None):
            self.http_auth_level = http_auth_level
            self.routes = {}

        def route(self, route: str, methods=None):
            def decorator(fn):
                self.routes[(route, tuple(methods or ["GET"]))] = fn
                return fn

            return decorator

    mod = types.ModuleType("azure.functions")
    mod.AuthLevel = types.SimpleNamespace(FUNCTION="FUNCTION")
    mod.HttpRequest = FakeHttpRequest
    mod.HttpResponse = FakeHttpResponse
    mod.FunctionApp = FakeFunctionApp

    azure_pkg = sys.modules.get("azure") or types.ModuleType("azure")
    azure_pkg.functions = mod
    monkeypatch.setitem(sys.modules, "azure", azure_pkg)
    monkeypatch.setitem(sys.modules, "azure.functions", mod)
    return FakeHttpRequest


def _load_function_app():
    path = Path(__file__).resolve().parents[1] / "function_app" / "function_app.py"
    spec = importlib.util.spec_from_file_location("test_fn_app_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_gold(root: Path, as_of: str = "2026-11-20") -> None:
    sink = LocalSink(root)
    predictions = pd.DataFrame([{
        "as_of": as_of,
        "run_id": "run123",
        "model_vol": "garch",
        "model_ret": "expanding_mean",
        "base_date": as_of,
        "last_close": 7200.0,
        "horizon_step": 1,
        "target_date": "2026-11-23",
        "forecast_variance_pct2": 1.44,
        "forecast_volatility_pct": 1.2,
        "cumulative_variance_pct2": 1.44,
        "annualized_volatility_pct": 19.0,
        "risk_regime": "NORMAL",
        "prob_high": 0.35,
        "tau": 1.8,
        "persistence": 0.94,
        "ml_return_enabled": False,
        "expected_log_return_pct": 0.02,
        "cumulative_expected_return_pct": 0.02,
        "median_close_price": 7201.44,
        "price_interval_low": 7030.0,
        "price_interval_high": 7370.0,
        "prob_up_vs_last_close": 0.51,
        "direction_signal": "NO_SIGNAL",
    }])
    assert predictions.columns.tolist() == list(GOLD_PREDICTIONS.columns)
    sink.write_parquet(predictions, dataset.as_of_key(dataset.GOLD_PREDICTIONS, as_of))

    params = pd.DataFrame([{
        "as_of": as_of,
        "c_train": 1.4,
        "tau": 1.8,
        "long_run_var_pct2": 1.2,
        "persistence": 0.94,
        "n_train": 1200,
        "n_val": 400,
        "n_test": 400,
        "i_tr": 1200,
        "i_va": 1600,
        "n_real": 2000,
        "n_features": 48,
        "window": 22,
        "data_fingerprint": "fp123",
    }])
    assert params.columns.tolist() == list(GOLD_PARAMS.columns)
    sink.write_parquet(params, dataset.as_of_key(dataset.GOLD_PARAMS, as_of))

    metrics = pd.DataFrame([{
        "as_of": as_of,
        "run_id": "run123",
        "split": "test",
        "model": "garch",
        "metric": "qlike",
        "horizon": 1,
        "value": 0.52,
    }])
    assert metrics.columns.tolist() == list(GOLD_METRICS.columns)
    sink.write_parquet(metrics, dataset.as_of_key(dataset.GOLD_METRICS, as_of))

    published = pd.DataFrame([{
        "as_of": as_of,
        "run_id": "run123",
        "model_vol": "garch",
        "horizons": 1,
        "ok": True,
    }])
    assert published.columns.tolist() == list(GOLD_PUBLISHED.columns)
    sink.write_parquet(published, dataset.as_of_key(dataset.GOLD_PUBLISHED, as_of))

    append_run(
        sink,
        as_of=as_of,
        run_id="run123",
        stage="train",
        trigger="manual",
        promoted=True,
        val_metric=0.27,
        test_metric=None,
        duration_s=2.5,
    )


def test_function_bundle_imports_without_heavy_ml_deps(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    bundle_script = repo_root / "function_app" / "bundle.sh"
    subprocess.run(["bash", os.fspath(bundle_script)], check=True)

    fn_dir = repo_root / "function_app"
    assert (fn_dir / "ihsg" / "serving" / "handler.py").is_file()
    assert not (fn_dir / "ihsg" / "models").exists()
    assert not (fn_dir / "ihsg" / "pipelines").exists()

    env = {
        **os.environ,
        "PYTHONPATH": os.fspath(fn_dir),
        "SINK": "local",
        "LOCAL_ROOT": os.fspath(tmp_path),
    }
    code = (
        "import sys; "
        "from ihsg.config import Settings; "
        "from ihsg.runlog import read_runs; "
        "from ihsg.serving import handler; "
        "from ihsg.storage import get_sink; "
        "assert 'torch' not in sys.modules; "
        "assert 'arch' not in sys.modules; "
        "assert 'prefect' not in sys.modules; "
        "assert 'streamlit' not in sys.modules; "
        "print(handler.health(Settings.from_env())['status'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.fspath(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "empty"


def test_function_app_routes(monkeypatch, tmp_path):
    FakeHttpRequest = _install_fake_azure_functions(monkeypatch)
    monkeypatch.setenv("SINK", "local")
    monkeypatch.setenv("LOCAL_ROOT", os.fspath(tmp_path))
    _seed_gold(tmp_path, "2026-11-20")

    fn_mod = _load_function_app()

    health_resp = fn_mod.health(FakeHttpRequest())
    assert health_resp.status_code == 200
    assert json.loads(health_resp.body)["latest_as_of"] == "2026-11-20"

    latest_resp = fn_mod.forecast_latest(FakeHttpRequest())
    assert latest_resp.status_code == 200
    latest_body = json.loads(latest_resp.body)
    assert latest_body["model"]["volatility"] == "garch"
    assert latest_body["metadata"]["as_of"] == "2026-11-20"

    by_date_resp = fn_mod.forecast(FakeHttpRequest(params={"as_of": "2026-11-20"}))
    assert by_date_resp.status_code == 200

    runs_resp = fn_mod.runs(FakeHttpRequest(params={"limit": "5"}))
    assert runs_resp.status_code == 200
    runs_body = json.loads(runs_resp.body)
    assert len(runs_body["runs"]) == 1
    assert runs_body["runs"][0]["run_id"] == "run123"

    # /admin/retrain auth + ACA trigger
    assert fn_mod.retrain(FakeHttpRequest()).status_code == 503
    monkeypatch.setenv("ADMIN_TOKEN", "secret123")
    assert fn_mod.retrain(FakeHttpRequest()).status_code == 401
    assert fn_mod.retrain(FakeHttpRequest(headers={"X-Admin-Token": "secret123"})).status_code == 503

    called = {}

    def fake_trigger(job_id: str, api_version: str = "2024-03-01") -> dict:
        called["job_id"] = job_id
        called["api_version"] = api_version
        return {"id": "exec-1"}

    monkeypatch.setenv(
        "ACA_JOB_RESOURCE_ID",
        "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.App/jobs/job-ihsg-weekly",
    )
    monkeypatch.setattr(fn_mod, "_trigger_aca_job", fake_trigger)
    retrain_resp = fn_mod.retrain(FakeHttpRequest(headers={"X-Admin-Token": "secret123"}))
    assert retrain_resp.status_code == 202
    assert called["job_id"].endswith("/jobs/job-ihsg-weekly")
