"""Model deep learning: GRU multi-head (varians + probabilitas rezim).

Kenapa GRU dan bukan model tabular:
* input berupa **urutan** (window x fitur), bukan satu vektor tabular;
* satu model menghasilkan dua keluaran sekaligus:
    - ``log`` varians hari berikutnya (regresi, target ``log(c_train * parkinson)``),
    - logit probabilitas rezim HIGH (klasifikasi, untuk dashboard).

Determinisme:
* ``torch.manual_seed`` + ``torch.set_num_threads(1)``,
* tidak ada dropout saat evaluasi (``model.eval()``),
* bobot terbaik dipilih berdasarkan loss validasi, bukan epoch terakhir.

``torch`` diimpor secara lazy supaya paket ini tetap bisa diimpor (dan diuji)
tanpa torch terpasang.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - bergantung lingkungan
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

EPS = 1e-6


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch tidak tersedia. Pasang wheel CPU: "
            "pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1"
        )


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.10
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 15
    seed: int = 42
    classification_weight: float = 0.5


if TORCH_AVAILABLE:  # pragma: no branch

    class GRUForecaster(nn.Module):  # type: ignore[misc]
        """GRU -> dua head linear (log-varians, logit rezim HIGH)."""

        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head_variance = nn.Linear(hidden, 1)
            self.head_regime = nn.Linear(hidden, 1)

        def forward(self, x):  # noqa: ANN001, ANN201
            output, _ = self.gru(x)
            last = output[:, -1, :]
            return self.head_variance(last).squeeze(-1), self.head_regime(last).squeeze(-1)

else:  # pragma: no cover

    class GRUForecaster:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


def build_windows(
    features: np.ndarray,
    log_variance: np.ndarray,
    label_high: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Potong deret menjadi ``(X, y_var, y_regime, end_index)``.

    ``X[i]`` berisi baris ``t-window .. t-1`` dan menargetkan baris ``t``.
    Fitur pada baris ``t`` sendiri sudah ber-lag (informasi t-1), sehingga
    tidak ada kebocoran.
    """
    n = len(features)
    if n <= window:
        raise ValueError(f"data terlalu pendek: {n} baris untuk window {window}")

    # Jendela BERAKHIR pada baris target: fitur baris t sudah ber-lag (informasi
    # t-1), jadi untuk memprediksi target hari t kita pakai baris t itu sendiri.
    # Ini harus konsisten dengan predict(), yang memakai `scaled[-window:]`.
    ends = np.arange(window - 1, n)
    X = np.stack([features[end - window + 1 : end + 1] for end in ends]).astype("float32")
    y_var = log_variance[ends].astype("float32")
    y_regime = label_high[ends].astype("float32")
    valid = np.isfinite(y_var)
    return X[valid], y_var[valid], y_regime[valid], ends[valid]


def _seed_everything(seed: int) -> None:
    require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    try:
        # Hanya boleh dipanggil sekali per proses. Backfill menjalankan training
        # berkali-kali dalam satu proses, jadi kegagalan di sini harus diabaikan.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def fit(
    X_train: np.ndarray,
    y_var_train: np.ndarray,
    y_regime_train: np.ndarray,
    X_val: np.ndarray,
    y_var_val: np.ndarray,
    y_regime_val: np.ndarray,
    *,
    n_features: int,
    config: TrainConfig,
) -> dict:
    """Latih model deterministik dengan early stopping pada loss validasi."""
    require_torch()
    _seed_everything(config.seed)

    model = GRUForecaster(n_features, config.hidden, config.layers, config.dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    tensors = {
        "x": torch.from_numpy(X_train),
        "yv": torch.from_numpy(y_var_train),
        "yr": torch.from_numpy(y_regime_train),
        "xv": torch.from_numpy(X_val),
        "yvv": torch.from_numpy(y_var_val),
        "yrv": torch.from_numpy(y_regime_val),
    }

    best_state: dict | None = None
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(tensors["x"]))
        total = 0.0
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            optimizer.zero_grad()
            log_var, logit = model(tensors["x"][batch])
            loss = mse(log_var, tensors["yv"][batch]) + config.classification_weight * bce(
                logit, tensors["yr"][batch]
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        train_loss = total / max(len(permutation), 1)

        model.eval()
        with torch.no_grad():
            log_var_val, logit_val = model(tensors["xv"])
            val_loss = float(
                mse(log_var_val, tensors["yvv"]).item()
                + config.classification_weight * bce(logit_val, tensors["yrv"]).item()
            )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "history": history,
    }


def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(varians, probabilitas_HIGH)``."""
    require_torch()
    model.eval()
    with torch.no_grad():
        log_var, logit = model(torch.from_numpy(X.astype("float32")))
        variance = torch.exp(log_var).cpu().numpy().astype(float)
        probability = torch.sigmoid(logit).cpu().numpy().astype(float)
    return variance, probability


def dumps(model, config: TrainConfig, window: int, n_features: int, feature_names: list[str]) -> bytes:
    require_torch()
    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "window": window,
            "n_features": n_features,
            "feature_names": feature_names,
        },
        buffer,
    )
    return buffer.getvalue()


def loads(data: bytes) -> tuple[object, dict]:
    require_torch()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    config = payload["config"]
    model = GRUForecaster(
        payload["n_features"], config["hidden"], config["layers"], config["dropout"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {
        "window": payload["window"],
        "n_features": payload["n_features"],
        "feature_names": payload["feature_names"],
        "config": config,
    }
    return model, metadata
