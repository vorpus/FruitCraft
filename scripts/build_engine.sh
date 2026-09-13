#!/usr/bin/env bash
# Build the OpenSnowstorm engine bridge (_engine) into fruitcraft/.
# If the pysdl2-dll wheel is installed, the engine's SDL renderer is compiled
# in (runtime default stays headless; BroodWarGame/set_gui toggles the window).
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python
CMAKE=.venv/bin/cmake

UI_ARGS=()
SDL_DIR=$("$PYBIN" - <<'EOF' 2>/dev/null || true
import os, sdl2dll
print(os.path.join(os.path.dirname(sdl2dll.__file__), "dll"))
EOF
)
if [ -n "${SDL_DIR:-}" ] && [ -f "$SDL_DIR/libSDL2-2.0.so" ]; then
    echo "SDL renderer: ENABLED (lib from $SDL_DIR)"
    # upstream's OPENBW_NO_SDL_MIXER path doesn't compile; build with the real
    # mixer instead (header in bridge/sdl_compat, lib from the wheel)
    if grep -q "OPENBW_NO_SDL_MIXER" third_party/opensnowstorm/mini-openbwapi/CMakeLists.txt; then
        git -C third_party/opensnowstorm apply ../../patches/opensnowstorm-enable-mixer.patch
    fi
    # the wheel ships unversioned .so files; the linker records the SONAME
    ln -sf libSDL2-2.0.so "$SDL_DIR/libSDL2-2.0.so.0"
    ln -sf libSDL2_mixer-2.0.so "$SDL_DIR/libSDL2_mixer-2.0.so.0"
    # upstream's sdl2.cpp uses the mixer even with OPENBW_NO_SDL_MIXER set, so
    # supply the mixer header (bridge/sdl_compat) + the wheel's mixer lib too
    UI_ARGS=(
        -DFRUITCRAFT_UI=ON
        -DSDL2_INCLUDE_DIR="$(pwd)/third_party/opensnowstorm/SDL2-2.30.2/include;$(pwd)/bridge/sdl_compat"
        -DSDL2_LIBRARY="$SDL_DIR/libSDL2-2.0.so;$SDL_DIR/libSDL2_mixer-2.0.so"
        -DFRUITCRAFT_SDL_RPATH="$SDL_DIR"
    )
else
    echo "SDL renderer: disabled (pip install pysdl2-dll to enable)"
fi

"$CMAKE" -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$(pwd)/.venv/bin/ninja" \
    -Dpybind11_DIR="$("$PYBIN" -m pybind11 --cmakedir)" \
    -DPython_EXECUTABLE="$(pwd)/$PYBIN" \
    "${UI_ARGS[@]}"
"$CMAKE" --build build -j"$(nproc)"
echo "OK: $(ls fruitcraft/_engine*.so)"
