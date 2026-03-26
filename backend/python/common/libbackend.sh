#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTHON_PATCH="${PYTHON_PATCH:-18}"
PY_STANDALONE_TAG="${PY_STANDALONE_TAG:-20250818}"
PORTABLE_PYTHON="${PORTABLE_PYTHON:-false}"
USE_PIP="${USE_PIP:-true}"

backend_dir="${LOCALAI_BACKEND_DIR:-}"
if [ -z "$backend_dir" ]; then
    backend_dir="$(cd -- "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
fi

EDIR="$backend_dir"
BACKEND_NAME="${backend_dir##*/}"
MY_DIR="$backend_dir"
backend_root="$(cd -- "$backend_dir/../.." && pwd)"
BUILD_PROFILE="${BUILD_PROFILE:-${BUILD_TYPE:-cpu}}"

_portable_dir() {
    printf '%s\n' "$EDIR/python"
}

_portable_bin() {
    printf '%s\n' "$(_portable_dir)/bin"
}

_portable_python() {
    if [ -x "$(_portable_bin)/python3" ]; then
        printf '%s\n' "$(_portable_bin)/python3"
    else
        printf '%s\n' "$(_portable_bin)/python"
    fi
}

_triple() {
    local os arch libc
    case "$(uname -s)" in
        Linux*) os="unknown-linux" ;;
        Darwin*) os="apple-darwin" ;;
        *) echo "Unsupported OS $(uname -s)" >&2; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) echo "Unsupported arch $(uname -m)" >&2; exit 1 ;;
    esac

    if [ "$os" = "unknown-linux" ]; then
        libc="gnu"
        printf '%s\n' "${arch}-${os}-${libc}"
    else
        printf '%s\n' "${arch}-${os}"
    fi
}

ensurePortablePython() {
    local pdir pbin full_ver filename url archive inner inner_root pyvenv_cfg sed_expr

    pdir="$(_portable_dir)"
    pbin="$(_portable_bin)"
    if [ -x "$pbin/python3" ] || [ -x "$pbin/python" ]; then
        return 0
    fi

    mkdir -p "$pdir"
    full_ver="${PYTHON_VERSION}.${PYTHON_PATCH}"
    filename="cpython-${full_ver}+${PY_STANDALONE_TAG}-$(_triple)-install_only.tar.gz"
    url="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_STANDALONE_TAG}/${filename}"
    archive="$pdir/$filename"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 1 -o "$archive" "$url"
    else
        wget -O "$archive" "$url"
    fi

    tar -xzf "$archive" -C "$pdir"
    rm -f "$archive"

    inner="$(find "$pdir" -maxdepth 3 -type f -path '*/bin/python*' | head -n 1 || true)"
    if [ -n "$inner" ]; then
        inner_root="$(dirname "$(dirname "$inner")")"
        if [ "$inner_root" != "$pdir" ]; then
            mv "$inner_root"/* "$pdir"/
            rmdir "$inner_root" 2>/dev/null || true
        fi
    fi

    pyvenv_cfg="$pdir/pyvenv.cfg"
    if [ -f "$pyvenv_cfg" ]; then
        sed_expr="s|^home = .*|home = $pdir/bin|"
        sed -i "$sed_expr" "$pyvenv_cfg" 2>/dev/null || sed -i '' "$sed_expr" "$pyvenv_cfg"
    fi

    "$(_portable_python)" -V >/dev/null
}

_makeVenvPortable() {
    local venv_dir vbin pyvenv_cfg portable_dir sed_expr f

    venv_dir="$EDIR/venv"
    vbin="$venv_dir/bin"
    [ -d "$vbin" ] || return 0

    rm -f "$vbin/python3" "$vbin/python"
    ln -s ../../python/bin/python3 "$vbin/python3"
    ln -s python3 "$vbin/python"

    pyvenv_cfg="$venv_dir/pyvenv.cfg"
    if [ -f "$pyvenv_cfg" ]; then
        portable_dir="$(_portable_dir)"
        portable_dir="$(cd "$portable_dir" && pwd)"
        sed_expr="s|^home = .*|home = $portable_dir/bin|"
        sed -i "$sed_expr" "$pyvenv_cfg" 2>/dev/null || sed -i '' "$sed_expr" "$pyvenv_cfg"
    fi

    for f in "$vbin"/*; do
        [ -f "$f" ] || continue
        head -c2 "$f" 2>/dev/null | grep -q '^#!' || continue
        if head -n1 "$f" | grep -Fq "$venv_dir"; then
            sed -i '1s|^#!.*$|#!/usr/bin/env python3|' "$f" 2>/dev/null || sed -i '' '1s|^#!.*$|#!/usr/bin/env python3|' "$f"
            chmod +x "$f" 2>/dev/null || true
        fi
    done
}

detect_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi

    if [ "$PORTABLE_PYTHON" = "true" ] || [ -x "$(_portable_python)" ]; then
        ensurePortablePython
        printf '%s\n' "$(_portable_python)"
        return 0
    fi

    if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        printf 'python%s\n' "$PYTHON_VERSION"
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
    local interpreter

    if [ -x "$EDIR/venv/Scripts/python.exe" ]; then
        return 0
    fi

    if [ -x "$EDIR/venv/bin/python" ]; then
        if [ "$PORTABLE_PYTHON" = "true" ] || [ -x "$(_portable_python)" ]; then
            _makeVenvPortable
        fi
        return 0
    fi

    interpreter="$(detect_python)"
    "$interpreter" -m venv --copies "$EDIR/venv"

    if [ -x "$EDIR/venv/bin/python" ]; then
        "$EDIR/venv/bin/python" -m pip install --upgrade pip
        if [ "$PORTABLE_PYTHON" = "true" ] || [ -x "$(_portable_python)" ]; then
            _makeVenvPortable
        fi
    else
        "$EDIR/venv/Scripts/python.exe" -m pip install --upgrade pip
    fi
}

_venv_python() {
    if [ -x "$EDIR/venv/bin/python" ]; then
        printf '%s\n' "$EDIR/venv/bin/python"
    else
        printf '%s\n' "$EDIR/venv/Scripts/python.exe"
    fi
}

_venv_pip() {
    if [ -x "$EDIR/venv/bin/pip" ]; then
        printf '%s\n' "$EDIR/venv/bin/pip"
    else
        printf '%s\n' "$EDIR/venv/Scripts/pip.exe"
    fi
}

_pip_install() {
    "$(_venv_python)" -m pip install "$@"
}

runProtogen() {
    ensureVenv
    _pip_install grpcio-tools
    pushd "$EDIR" >/dev/null
    "$(_venv_python)" -m grpc_tools.protoc -I../../ -I./ --python_out=. --grpc_python_out=. backend.proto
    popd >/dev/null
}

installRequirements() {
    local req profile_req

    ensureVenv
    for req in "$EDIR/requirements-install.txt" "$EDIR/requirements.txt"; do
        if [ -f "$req" ]; then
            _pip_install ${EXTRA_PIP_INSTALL_FLAGS:-} -r "$req"
        fi
    done

    profile_req="$EDIR/requirements-${BUILD_PROFILE}.txt"
    if [ -f "$profile_req" ]; then
        _pip_install ${EXTRA_PIP_INSTALL_FLAGS:-} -r "$profile_req"
    fi

    runProtogen
}

startBackend() {
    ensureVenv
    if [ ! -f "$EDIR/backend_pb2.py" ] || [ ! -f "$EDIR/backend_pb2_grpc.py" ]; then
        runProtogen
    fi
    exec "$(_venv_python)" "$MY_DIR/backend.py" "$@"
}

runUnittests() {
    ensureVenv
    if [ ! -f "$EDIR/backend_pb2.py" ] || [ ! -f "$EDIR/backend_pb2_grpc.py" ]; then
        runProtogen
    fi
    pushd "$MY_DIR" >/dev/null
    "$(_venv_python)" -m unittest test.py
    popd >/dev/null
}
