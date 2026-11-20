#!/bin/sh
# =============================================================================
# Idempotent Garage bootstrap.
#
# Membuat bucket medallion dan memberi akses key aplikasi. Memakai Admin API v2
# (Garage >= 2.x). Endpoint /v1/ dari tutorial lama TIDAK berlaku.
#
# Aman dijalankan berkali-kali: CreateBucket yang gagal (bucket sudah ada)
# ditangani dengan GetBucketInfo.
#
# Env:
#   GARAGE_ADMIN_TOKEN         bearer token Admin API
#   GARAGE_DEFAULT_ACCESS_KEY  access key id yang diberi izin
#   GARAGE_BUCKETS             daftar bucket, default "bronze silver gold mlartifacts"
# =============================================================================
set -eu

API="${GARAGE_ADMIN_API:-http://garage:3903}"
AUTH="Authorization: Bearer ${GARAGE_ADMIN_TOKEN:?GARAGE_ADMIN_TOKEN is required}"
KEY="${GARAGE_DEFAULT_ACCESS_KEY:?GARAGE_DEFAULT_ACCESS_KEY is required}"
BUCKETS="${GARAGE_BUCKETS:-bronze silver gold mlartifacts}"

echo "garage-init: menunggu Admin API di ${API} ..."
i=0
until curl -fsS -H "$AUTH" "$API/v2/ListBuckets" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 60 ]; then
    echo "garage-init: Admin API tidak siap setelah 120 detik" >&2
    exit 1
  fi
  sleep 2
done

for bucket in $BUCKETS; do
  echo "garage-init: memastikan bucket '${bucket}'"

  # CreateBucket mengembalikan error bila bucket sudah ada -> abaikan.
  create_response=$(curl -sS -X POST "$API/v2/CreateBucket" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"globalAlias\":\"${bucket}\"}" 2>/dev/null || true)

  bucket_id=$(printf '%s' "$create_response" | jq -r '.id // empty' 2>/dev/null || true)

  if [ -z "$bucket_id" ]; then
    bucket_id=$(curl -sS -G "$API/v2/GetBucketInfo" \
      -H "$AUTH" --data-urlencode "globalAlias=${bucket}" |
      jq -r '.id // empty')
  fi

  if [ -z "$bucket_id" ]; then
    echo "garage-init: tidak bisa menentukan id bucket '${bucket}'" >&2
    exit 1
  fi

  curl -fsS -X POST "$API/v2/AllowBucketKey" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"bucketId\":\"${bucket_id}\",\"accessKeyId\":\"${KEY}\",\"permissions\":{\"read\":true,\"write\":true,\"owner\":true}}" \
    >/dev/null

  echo "garage-init: '${bucket}' siap (id=${bucket_id})"
done

echo "garage-init: selesai. Bucket: ${BUCKETS}"
