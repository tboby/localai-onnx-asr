#!/bin/bash
set -euo pipefail

backend_dir="${LOCALAI_BACKEND_DIR:-}"
if [ -z "$backend_dir" ]; then
    backend_dir="$(cd -- "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
fi
backend_root="$(cd -- "$backend_dir/../.." && pwd)"
venv_dir="${VENV_DIR:-$backend_dir/venv}"

venv_python=""
venv_pip=""

detect_python() {
    if [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        printf 'python\n'
        return 0
    fi

    printf 'Unable to find a Python interpreter.\n' >&2
    return 1
}

ensureVenv() {
    local host_python

    host_python="$(detect_python)"
    if [ ! -d "$venv_dir" ]; then
        "$host_python" -m venv "$venv_dir"
    fi

    if [ -x "$venv_dir/Scripts/python.exe" ]; then
        venv_python="$venv_dir/Scripts/python.exe"
        venv_pip="$venv_dir/Scripts/pip.exe"
    else
        venv_python="$venv_dir/bin/python"
        venv_pip="$venv_dir/bin/pip"
    fi
}

runProtogen() {
    ensureVenv
    "$venv_python" -m grpc_tools.protoc \
        -I"$backend_root" \
        --python_out="$backend_dir" \
        --grpc_python_out="$backend_dir" \
        "$backend_root/backend.proto"
}

installRequirements() {
    local profile profile_file

    ensureVenv
    "$venv_python" -m pip install --upgrade pip
    "$venv_pip" install ${EXTRA_PIP_INSTALL_FLAGS:-} -r "$backend_dir/requirements.txt"

    profile="${BUILD_PROFILE:-cpu}"
    profile_file="$backend_dir/requirements-$profile.txt"
    if [ -f "$profile_file" ]; then
        "$venv_pip" install ${EXTRA_PIP_INSTALL_FLAGS:-} -r "$profile_file"
    fi

    runProtogen
}

startBackend() {
    ensureVenv
    if [ ! -f "$backend_dir/backend_pb2.py" ] || [ ! -f "$backend_dir/backend_pb2_grpc.py" ]; then
        runProtogen
    fi
    exec "$venv_python" "$backend_dir/backend.py" "$@"
}

runUnittests() {
    ensureVenv
    if [ ! -f "$backend_dir/backend_pb2.py" ] || [ ! -f "$backend_dir/backend_pb2_grpc.py" ]; then
        runProtogen
    fi
    exec "$venv_python" "$backend_dir/test.py"
}
