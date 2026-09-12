#!/usr/bin/env bash
# Build the OpenSnowstorm engine bridge (_engine) into fruitcraft/.
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python
CMAKE=.venv/bin/cmake
"$CMAKE" -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$(pwd)/.venv/bin/ninja" \
    -Dpybind11_DIR="$("$PYBIN" -m pybind11 --cmakedir)" \
    -DPython_EXECUTABLE="$(pwd)/$PYBIN"
"$CMAKE" --build build -j"$(nproc)"
echo "OK: $(ls fruitcraft/_engine*.so)"
