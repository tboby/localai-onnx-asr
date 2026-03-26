#!/bin/bash
set -euo pipefail

backend_dir="/opt/localai/backend/python/onnx-asr"
addr="${LOCALAI_GRPC_ADDRESS:-0.0.0.0:${PORT:-50051}}"

exec bash "$backend_dir/run.sh" --addr "$addr" "$@"
