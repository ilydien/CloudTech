"""Dashboard Streamlit: hasil olahan ML, bukan statistik deskriptif.

Isi:
1. kartu ringkasan (as_of, model champion, rezim, probabilitas HIGH),
2. tabel ramalan 1..7 hari + interval 95%,
3. grafik volatilitas aktual vs GRU vs HAR pada periode uji (gold/vol_eval),
4. klasifikasi rezim HIGH vs NORMAL pada periode uji (gold/regime_eval),
5. riwayat run/retrain (gold/run_log) - bukti retrain berulang.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ihsg import dataset
from ihsg.config import Settings
from ihsg.storage import get_sink

st.set_page_config(page_title="IHSG Forecast Dashboard", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def _sink():
    return get_sink(_settings())


@st.cache_data(ttl=60)
def _available_as_of() -> list[str]:
    return dataset.existing_as_of(_sink(), dataset.GOLD_PREDICTIONS)


@st.cache_data(ttl=60)
def _load(as_of: str) -> dict[str, pd.DataFrame]:
    sink = _sink()
    tables = {
        "predictions": dataset.GOLD_PREDICTIONS,
        "regime": dataset.GOLD_REGIME,
        "vol_eval": dataset.GOLD_VOL_EVAL,
        "regime_eval": dataset.GOLD_REGIME_EVAL,
        "metrics": dataset.GOLD_METRICS,
    }
    out: dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        frame = sink.try_read_parquet(dataset.as_of_key(table, as_of))
        out[name] = pd.DataFrame() if frame is None else frame
    champion = sink.try_read_parquet(dataset.GOLD_CHAMPION)
    out["champion"] = pd.DataFrame() if champion is None else champion
    return out


@st.cache_data(ttl=60)
def _run_log() -> pd.DataFrame:
    from ihsg.runlog import read_runs

    return read_runs(_sink())


st.title("IHSG - Volatilitas & Return (hasil model)")

available = _available_as_of()
if not available:
    st.warning(
        "Belum ada gold/predictions. Jalankan dulu:\n\n"
        "```\ndocker compose run --rm engine run --retrain --as-of auto\n```"
    )
    st.stop()

as_of = st.sidebar.selectbox("Tanggal acuan (as_of)", options=list(reversed(available)))
if st.sidebar.button("Muat ulang"):
    st.cache_data.clear()

data = _load(as_of)
predictions = data["predictions"]
champion = data["champion"]

if predictions.empty:
    st.error(f"gold/predictions kosong untuk {as_of}")
    st.stop()

first = predictions.sort_values("horizon_step").iloc[0]
model_vol = str(champion["model_vol"].iloc[0]) if not champion.empty else "?"

col1, col2, col3, col4 = st.columns(4)
col1.metric("As of", as_of)
col2.metric("Model volatilitas", model_vol.upper())
col3.metric("Volatilitas H+1", f"{first.forecast_volatility_pct:.2f}%")
col4.metric(
    "Probabilitas rezim HIGH",
    f"{first.prob_high:.0%}" if pd.notna(first.prob_high) else "n/a",
    help="Keluaran head klasifikasi model GRU untuk horizon 1 hari.",
)

st.subheader("Ramalan 1-7 hari bursa")
view = predictions.sort_values("horizon_step").copy()
view["interval_95"] = (
    view["price_interval_low"].map(lambda x: f"{x:,.0f}")
    + " - "
    + view["price_interval_high"].map(lambda x: f"{x:,.0f}")
)
st.dataframe(
    view[
        ["horizon_step", "target_date", "forecast_volatility_pct", "annualized_volatility_pct",
         "risk_regime", "prob_high", "cumulative_expected_return_pct", "median_close_price",
         "interval_95", "direction_signal"]
    ].rename(columns={
        "horizon_step": "H+",
        "target_date": "Tanggal",
        "forecast_volatility_pct": "Vol (%)",
        "annualized_volatility_pct": "Vol tahunan (%)",
        "risk_regime": "Rezim",
        "prob_high": "P(HIGH)",
        "cumulative_expected_return_pct": "Return kumulatif (%)",
        "median_close_price": "Median close",
        "interval_95": "Interval 95%",
        "direction_signal": "Sinyal arah",
    }),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)

with left:
    st.subheader("Volatilitas aktual vs model (periode uji)")
    vol_eval = data["vol_eval"]
    if vol_eval.empty:
        st.info("gold/vol_eval belum tersedia untuk tanggal ini.")
    else:
        chart = vol_eval.set_index("date")[["actual_var", "gru_var", "har_var"]].copy()
        for column in chart.columns:
            chart[column] = chart[column] ** 0.5
        st.line_chart(chart.rename(columns={
            "actual_var": "Aktual", "gru_var": "GRU", "har_var": "HAR",
        }), height=320)
        st.caption("Akar varians harian (%) - proxy Parkinson terskala.")

with right:
    st.subheader("Klasifikasi rezim HIGH")
    regime_eval = data["regime_eval"]
    if regime_eval.empty:
        st.info("gold/regime_eval belum tersedia untuk tanggal ini.")
    else:
        accuracy = float(regime_eval["correct"].mean())
        base_rate = float(regime_eval["actual_high"].mean())
        st.metric("Akurasi klasifikasi", f"{accuracy:.1%}")
        st.caption(
            f"Base rate hari HIGH = {base_rate:.1%}. "
            "Akurasi tinggi saja tidak cukup: lihat recall & AUC di tabel metrik."
        )
        recent = regime_eval.sort_values("date").set_index("date")[["prob_high"]].tail(60)
        st.area_chart(recent, height=200)

st.subheader("Metrik evaluasi")
metrics_frame = data["metrics"]
if metrics_frame.empty:
    st.info("gold/metrics belum tersedia.")
else:
    pivot = metrics_frame.pivot_table(
        index="model", columns=["split", "metric"], values="value", aggfunc="first"
    ).round(4)
    st.dataframe(pivot, use_container_width=True)

st.subheader("Riwayat run / retrain")
runs = _run_log()
if runs.empty:
    st.info("gold/run_log belum tersedia.")
else:
    st.dataframe(
        runs.sort_values("as_of", ascending=False)[
            ["as_of", "stage", "trigger", "promoted", "duration_s", "git_sha"]
        ].rename(columns={
            "as_of": "As of", "stage": "Tahap", "trigger": "Trigger",
            "promoted": "Dipromosikan", "duration_s": "Durasi (s)", "git_sha": "Git",
        }).head(12),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Total retrain tercatat: {len(runs)}")
