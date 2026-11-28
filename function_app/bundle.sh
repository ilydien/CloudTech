#!/usr/bin/env bash
# Salin hanya modul serving minimal dari ../ihsg ke function_app/ihsg
# agar Azure Function (Flex Consumption) bisa di-publish tanpa torch/arch/prefect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/ihsg"
DST="$SCRIPT_DIR/ihsg"

rm -rf "$DST" "$SCRIPT_DIR/__pycache__"
mkdir -p "$DST/serving"

FILES=(
  "__init__.py"
  "config.py"
  "dataset.py"
  "storage.py"
  "schemas.py"
  "runlog.py"
  "serving/__init__.py"
  "serving/handler.py"
)

for rel in "${FILES[@]}"; do
  cp "$SRC/$rel" "$DST/$rel"
done

echo "[func-bundle] disalin ${#FILES[@]} file ke $DST (tanpa torch/arch/prefect/streamlit)"
