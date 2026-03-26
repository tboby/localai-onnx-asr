#!/bin/bash
set -e

backend_dir="$(cd -- "$(dirname "$0")" && pwd)"
export LOCALAI_BACKEND_DIR="$backend_dir"

if [ -d "$backend_dir/common" ]; then
    source "$backend_dir/common/libbackend.sh"
else
    source "$backend_dir/../common/libbackend.sh"
fi

runUnittests
