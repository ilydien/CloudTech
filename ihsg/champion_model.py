"""Model pyfunc untuk MLflow Model Registry.

Champion di sistem ini bukan selalu satu objek model terlatih: bisa GARCH, HAR,
atau GRU. Yang benar-benar dipakai produksi adalah **resep**: model mana,
ramalan varians h=1, target mean reversion, dan persistence. Kelas ini
membungkus resep itu menjadi model MLflow yang bisa di-*version* dan dimuat
ulang, sehingga setiap promosi menghasilkan satu versi yang dapat direproduksi.

Modul ini mengimpor ``mlflow`` di level modul, jadi hanya boleh diimpor secara
lazy dari ``ihsg.tracking`` supaya `import ihsg` tetap ringan.
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

FLAT_MODELS = ("hist", "ewma")


class ChampionRecipe(mlflow.pyfunc.PythonModel):
    """Resep ramalan varians 1..H hari yang dipakai produksi."""

    def __init__(self, recipe: dict) -> None:
        self.recipe = dict(recipe)

    def predict(self, context, model_input=None, params=None) -> pd.DataFrame:  # noqa: ANN001
        recipe = self.recipe
        horizons = int(recipe["horizons"])
        h1 = float(recipe["h1_var"])

        if recipe["model"] in FLAT_MODELS:
            paths = [h1] * horizons
        else:
            uncond = float(recipe["uncond"])
            persistence = float(np.clip(recipe["persistence"], 0.0, 0.9999))
            paths = [
                uncond + persistence ** (k - 1) * (h1 - uncond)
                for k in range(1, horizons + 1)
            ]

        return pd.DataFrame(
            {
                "horizon_step": list(range(1, horizons + 1)),
                "forecast_variance_pct2": paths,
                "forecast_volatility_pct": [float(np.sqrt(value)) for value in paths],
                "model_vol": recipe["model"],
                "as_of": recipe["as_of"],
                "run_id": recipe["run_id"],
            }
        )
