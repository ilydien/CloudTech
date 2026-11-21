# Image MLflow dengan dukungan artefak S3 (Garage).
# Image resmi tidak menyertakan boto3, sehingga `--artifacts-destination s3://...`
# akan gagal tanpa tambahan ini.
FROM ghcr.io/mlflow/mlflow:v2.22.0

RUN pip install --no-cache-dir boto3
