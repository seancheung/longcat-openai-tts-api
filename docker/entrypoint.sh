#!/usr/bin/env bash
set -euo pipefail

: "${LONGCAT_MODEL:=meituan-longcat/LongCat-AudioDiT-1B}"
: "${LONGCAT_VOICES_DIR:=/voices}"
: "${LONGCAT_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export LONGCAT_MODEL LONGCAT_VOICES_DIR LONGCAT_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
