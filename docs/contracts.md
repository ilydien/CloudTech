# Kontrak tabel (medallion)

Dokumen ini adalah spesifikasi skema. Sumber kebenaran di kode:
[`ihsg/schemas.py`](../ihsg/schemas.py) — setiap penulisan divalidasi sebelum
`write_parquet`, jadi skema tidak bisa “melenceng” diam-diam.

Konvensi umum:

- semua file bernama `data.parquet`, kompresi `snappy`;
- tanggal selalu kolom (`date`, `base_date`, `as_of`), bukan index;
- partisi memakai Hive-style: `k=v/`, dan `=` hanya dipakai untuk partisi
  (karena itu `key` instrumen disanitasi: `^JKSE` → `JKSE`);
- `as_of` = tanggal acuan (hari bursa terakhir yang datanya dipakai);
- tidak ada CSV/JSON sebagai media penyimpanan.

---

## BRONZE

### `bronze/yf/prices/ticker=<K>/ingest_date=<D>/data.parquet`

Nilai mentah dari yfinance dengan `auto_adjust=False`. Nama kolom **tidak**
diubah.

| Kolom | Tipe | Null | Keterangan |
|---|---|---|---|
| `Date` | datetime64 | tidak | tanggal perdagangan (kalender instrumen) |
| `Open` | float64 | boleh | harga pembukaan |
| `High` | float64 | boleh | tertinggi |
| `Low` | float64 | boleh | terendah |
| `Close` | float64 | tidak | penutupan (belum disesuaikan) |
| `Adj Close` | float64 | tidak | penyesuaian dividen/split dari provider |
| `Volume` | float64 | boleh | 0 untuk instrumen kurs |

Kunci natural: `(Date)`.

### `bronze/yf/actions/ticker=<K>/ingest_date=<D>/data.parquet`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Date` | datetime64 | tanggal aksi korporasi |
| `Dividends` | float64 | dividen per saham (0 bila tidak ada) |
| `Stock Splits` | float64 | rasio split (0 bila tidak ada) |

### `bronze/yf/_manifest/ingest_date=<D>/data.parquet`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `ingest_date` | str | sama dengan partisi |
| `ticker` | str | simbol yfinance |
| `ticker_key` | str | key aman partisi |
| `rows` | int | jumlah baris yang ditulis |
| `date_min`, `date_max` | str | rentang tanggal data |
| `fingerprint` | str | SHA-256 isi DataFrame (lineage) |
| `provider`, `provider_version` | str | `yfinance`, versi paket |
| `request_start`, `request_end` | str | parameter unduhan (end eksklusif) |

---

## SILVER

### `silver/prices/ticker=<K>/data.parquet`

| Kolom | Tipe | Null | Keterangan |
|---|---|---|---|
| `date` | datetime64 | tidak | tanggal |
| `ticker` | str | tidak | simbol asli |
| `open`, `high`, `low`, `close` | float64 | boleh | harga mentah |
| `adj_close` | float64 | tidak | harga disesuaikan |
| `volume` | float64 | boleh | volume |
| `ret_cc_pct` | float64 | boleh | `100·ln(adj_close_t / adj_close_{t-1})` |
| `park_var` | float64 | boleh | varians Parkinson (%²) |
| `source_ingest_date` | str | tidak | partisi bronze asal (lineage) |

Kunci natural: `(ticker, date)`.

### `silver/panel/as_of=<D>/data.parquet`

Panel lebar pada kalender bursa BEI. Untuk setiap instrumen `K`:

| Kolom | Keterangan |
|---|---|
| `date` | hari bursa BEI |
| `{K}_open`, `{K}_high`, `{K}_low`, `{K}_close`, `{K}_adj_close` | harga setelah `ffill(limit=FFILL_LIMIT)` |
| `{K}_volume` | volume |
| `{K}_ret` | log-return % pada kalender instrumen sendiri |
| `{K}_park` | varians Parkinson %² |
| `is_filled_{K}` | `true` bila baris diisi oleh `ffill` |

Kunci natural: `(date)`.

### `silver/_dq/as_of=<D>/data.parquet`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `as_of` | str | tanggal acuan |
| `layer` | str | `silver` |
| `table` | str | mis. `silver.prices[JKSE]` |
| `check_name` | str | `not_null`, `positive_close`, `calendar_gaps`, `freshness`, … |
| `severity` | str | `hard` (menghentikan run) / `soft` / `info` |
| `passed` | bool | `failed_count == 0` |
| `failed_count`, `total_count` | int | agregat |
| `details` | str | format `k=v;k=v` (**bukan** JSON) |

Kunci: `(as_of, layer, table, check_name)`.

### `silver/_quarantine/as_of=<D>/data.parquet`

| Kolom | Keterangan |
|---|---|
| `as_of`, `layer`, `table` | konteks |
| `reason` | mis. `invalid_ohlc` |
| `key_value` | nilai kunci baris (tanggal) |
| `column_name`, `column_value` | kolom & nilai yang bermasalah (string) |

---

## GOLD

### `gold/features/as_of=<D>/data.parquet`

Satu baris per hari bursa, **plus satu baris terakhir untuk hari bursa berikutnya**
(fitur lengkap, target `null`).

Fitur per instrumen `K` (semua `shift(1)`, hanya informasi `t-1`):

| Kolom | Keterangan |
|---|---|
| `{K}_ret_lag1` | log-return kemarin (%) |
| `{K}_park_var_lag1` | varians Parkinson kemarin (%²) |
| `{K}_ret_ma5_lag1` | rata-rata return 5 hari |
| `{K}_rv5_lag1` | rata-rata return kuadrat 5 hari |
| `{K}_vol_chg_lag1` | `ln((V+1)/(V_{-1}+1))`; **dibuang** bila volume hampir selalu 0 (mis. `IDRX`) |

Fitur khusus IHSG:

| Kolom | Keterangan |
|---|---|
| `JKSE_sq_ret_lag1` | return kemarin dikuadratkan |
| `JKSE_rv22_lag1` | rata-rata return kuadrat 22 hari |
| `JKSE_park_ma5_lag1`, `JKSE_park_ma22_lag1` | rata-rata Parkinson 5 & 22 hari |

Target & harga:

| Kolom | Keterangan |
|---|---|
| `ret` | log-return IHSG hari itu (%) |
| `park` | varians Parkinson IHSG hari itu (%²) |
| `close`, `prev_close` | harga penutupan (display, bukan target) |

Kunci: `(date)`.

### `gold/params/as_of=<D>/data.parquet`

| Kolom | Keterangan |
|---|---|
| `c_train` | `mean(ret²)/mean(park)` **hanya dari data latih** |
| `tau` | kuantil `TAU_QUANTILE` dari `sqrt(c_train·park)` di data latih (ambang rezim HIGH) |
| `long_run_var_pct2` | rata-rata `c_train·park` di data latih (target mean reversion) |
| `persistence` | koefisien mean reversion (`EWMA_LAMBDA` default 0.94) |
| `n_train`, `n_val`, `n_test`, `i_tr`, `i_va`, `n_real` | batas split |
| `n_features`, `data_fingerprint` | jumlah fitur & hash tabel |
| `window` | panjang jendela sequence (dipakai payload) |

### `gold/predictions/as_of=<D>/data.parquet`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `as_of`, `run_id`, `model_vol`, `model_ret`, `base_date` | str | identitas run |
| `last_close` | float | penutupan terakhir |
| `horizon_step` | int | 1..H |
| `target_date` | str | tanggal target (kalender bursa) |
| `forecast_variance_pct2` | float | varians (%²) |
| `forecast_volatility_pct` | float | akar varians (%) |
| `cumulative_variance_pct2` | float | jumlah varians dari H+1 sampai H+k |
| `annualized_volatility_pct` | float | `vol·√252` |
| `risk_regime` | str | `HIGH` bila `vol > tau` |
| `prob_high` | float | keluaran head klasifikasi GRU (selalu dari GRU, apa pun model variansnya) |
| `tau` | float | ambang rezim yang dipakai (dibekukan di baris ini, bukan dibaca dari champion) |
| `persistence` | float | koefisien mean reversion model terpilih |
| `ml_return_enabled` | bool | apakah model return dipakai (gate lulus) |
| `expected_log_return_pct` | float | return harian yang diharapkan |
| `cumulative_expected_return_pct` | float | akumulasi |
| `median_close_price` | float | `last_close·exp(cum/100)` |
| `price_interval_low`, `price_interval_high` | float | interval 95% (lognormal) |
| `prob_up_vs_last_close` | float | `Φ(cum_return / cum_sd)` |
| `direction_signal` | str | `UP`/`DOWN` hanya bila gate model return lolos; selain itu `NO_SIGNAL` |

Kunci: `(as_of, horizon_step)`.

### `gold/regime/as_of=<D>/data.parquet`

`as_of`, `run_id`, `horizon_step`, `target_date`, `prob_high`, `risk_regime`.

### `gold/candidates/as_of=<D>/data.parquet`

Semua kandidat model volatilitas pada satu run. `predict` **hanya** membaca tabel
ini, sehingga tidak ada train/serve skew dan prediksi tidak butuh torch.

| Kolom | Keterangan |
|---|---|
| `as_of`, `run_id`, `model` | kunci; `model` ∈ `gru`, `har`, `garch`, `ewma`, `hist` |
| `val_qlike`, `test_qlike` | QLIKE h=1 pada validasi & uji |
| `h1_var` | ramalan varians untuk sesi berikutnya (skala %²) |
| `uncond`, `persistence` | parameter mean reversion model tersebut |
| `prob_high` | hanya terisi untuk `gru` |
| `selected` | `true` untuk model dengan QLIKE validasi terendah |

### `gold/published/as_of=<D>/data.parquet`

Penanda bahwa prediksi sudah lengkap. `as_of`, `run_id`, `model_vol`,
`horizons`, `ok`. Handler menolak menyajikan `as_of` tanpa penanda ini.

### `gold/_dq/as_of=<D>/data.parquet`

Kualitas fitur: kelengkapan fitur/target, jumlah baris yang dibuang saat warm-up.

### `gold/regime_eval/as_of=<D>/data.parquet`

Klasifikasi pada periode uji (bahan dashboard):

`as_of`, `run_id`, `date`, `prob_high`, `predicted_high`, `actual_high`, `correct`.

### `gold/vol_eval/as_of=<D>/data.parquet`

Perbandingan varians pada periode uji:

`as_of`, `run_id`, `date`, `actual_var`, `gru_var`, `har_var`, `ewma_var`, `hist_var`.

### `gold/metrics/as_of=<D>/data.parquet`

Format panjang:

`as_of`, `run_id`, `split` (`val`/`test`), `model`, `metric`, `horizon`, `value`.

`metric` ∈ `qlike`, `rmse_vol` (h=1..7), `dm_p_vs_har` (h=1..7), `dm_p_vs_garch`,
`r2_oos` (model `ridge`). Kunci: `(as_of, run_id, split, model, metric, horizon)`.

### `gold/run_log/as_of=<D>/stage=<S>/run_id=<R>/data.parquet`

| Kolom | Keterangan |
|---|---|
| `as_of`, `stage`, `run_id` | kunci; `stage` ∈ `silver`, `train`, `predict`. Satu entri = satu file, sehingga penulisan bersifat *upsert* tanpa baca-ubah-tulis |
| `trigger` | `manual` / `schedule` / `backfill` / `api` |
| `status` | `ok` / `failed` |
| `promoted` | apakah kandidat terbaik menggantikan champion |
| `val_metric`, `test_metric` | QLIKE model terpilih |
| `duration_s` | durasi (informasional) |
| `data_fingerprint`, `config_hash`, `git_sha` | lineage |

### `gold/champion/data.parquet`

Satu baris: `run_id`, `trained_as_of`, `model_vol`, `model_ret`, `val_qlike`,
`val_r2_oos`, `ml_return_enabled`, `tau`, `c_train`, `long_run_var_pct2`,
`persistence`, `window`, `horizons`, `n_features`, `config_hash`,
`data_fingerprint`, `mlflow_run_id`, `mlflow_version`.

Dua kolom terakhir menautkan champion ke satu versi di MLflow Model Registry
(kosong bila tracking dimatikan).

---

## ML ARTIFACTS

| File | Isi |
|---|---|
| `mlartifacts/run_id=<R>/model.pt` | `state_dict` GRU + konfigurasi + daftar fitur |
| `mlartifacts/run_id=<R>/scaler.parquet` | `feature`, `mean`, `scale` (urutan fitur ikut tersimpan) |
| `mlartifacts/run_id=<R>/ridge.joblib` | model return + daftar fitur |
| `mlartifacts/mlflow/<exp>/<run>/` | artefak run MLflow + bundel model registry |

`model.pt` hanya bisa dibuka dengan versi `torch`/`numpy` yang sama seperti di
`requirements.txt`. Naikkan versi → wajib retrain (`git_sha` + `config_hash`
tetap tercatat di `run_log`).

---

## Urutan ketergantungan

```
bronze (append-only)
   └─ silver/prices ─ silver/panel ─ silver/_dq, _quarantine
        └─ gold/features + gold/params
             └─ gold/artifacts (train) ─ gold/champion
                  └─ gold/predictions + gold/regime
                       └─ serving (JSON) / dashboard / Function
```

Menghapus `silver` dan `gold` tidak kehilangan apa pun selama `bronze` utuh —
seluruh turunan bisa dibangun ulang.
