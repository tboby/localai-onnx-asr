#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"
bundle_root="$script_dir"

if [ -d "$bundle_root/usr/local/bin" ]; then
    export PATH="$bundle_root/usr/local/bin:$bundle_root/usr/bin:$bundle_root/bin:${PATH:-}"
fi

library_paths=""
for candidate in \
    "$bundle_root/usr/local/lib" \
    "$bundle_root/usr/lib" \
    "$bundle_root/usr/lib/x86_64-linux-gnu" \
    "$bundle_root/lib" \
    "$bundle_root/lib/x86_64-linux-gnu"
do
    if [ -d "$candidate" ]; then
        if [ -n "$library_paths" ]; then
            library_paths="$library_paths:$candidate"
        else
            library_paths="$candidate"
        fi
    fi
done

if [ -n "$library_paths" ]; then
    export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
