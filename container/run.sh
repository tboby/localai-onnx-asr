#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"

if [ -d "$script_dir/opt/localai/backend/python/onnx-asr" ]; then
    backend_dir="$script_dir/opt/localai/backend/python/onnx-asr"
elif [ -d "/opt/localai/backend/python/onnx-asr" ]; then
    backend_dir="/opt/localai/backend/python/onnx-asr"
elif [ -d "$script_dir/backend/python/onnx-asr" ]; then
    backend_dir="$script_dir/backend/python/onnx-asr"
else
    printf 'Unable to locate packaged onnx-asr backend files.\n' >&2
    exit 1
fi

export LOCALAI_BACKEND_DIR="$backend_dir"
addr="${LOCALAI_GRPC_ADDRESS:-0.0.0.0:${PORT:-50051}}"

exec bash "$backend_dir/run.sh" --addr "$addr" "$@"
