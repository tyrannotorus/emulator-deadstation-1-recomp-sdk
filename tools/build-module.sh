#!/bin/sh
# Build the recomp module in the current folder, from your own disc, into build-brick/<id>/: what Deadstation installs.
#   sdk/tools/build-module.sh /path/to/disc.chd [ninja arguments]
# Every step runs without network (tools/sbx). PSXRecomp (sdk/psxrecomp) is brought to its audited pin and patches first
# (tools/prepare-psxrecomp.sh). From the disc come disc/ (the staged disc), generated/ (the game's C) and, if the module has
# captures in overlays/, those baked into generated/: never in git.
# Needs the toolchain (tools/toolchain/toolchain.sh; TG5040_ROOT if not in build-brick), the host's Vulkan headers,
# glslangValidator, gcc, clang, lld, Python 3, bubblewrap, the host's SDL3 and llvm-strip.
set -e
SDK=$(cd "$(dirname "$0")/.." && pwd)
MODULE=$PWD
CHD=${1:?usage: sdk/tools/build-module.sh DISC.chd [ninja arguments]}
shift
[ -f module.cfg ] && [ -f game.toml ] || { echo "build-module: no module.cfg and game.toml in this folder"; exit 2; }
M=$(sed -n 's/^id *= *//p' module.cfg)
EXEC=$(sed -n 's/^exec *= *//p' module.cfg)
TC=${TG5040_ROOT:-$PWD/build-brick/toolchain}
[ -d "$TC/aarch64-nextui-linux-gnu/libc" ] || { echo "build-module: no toolchain (tools/toolchain/toolchain.sh)"; exit 2; }
SBX="$SDK/tools/sbx"
JOBS=$(nproc)

"$SDK/tools/prepare-psxrecomp.sh" > /dev/null

# the translator (and libchdr), for this machine
mkdir -p build-brick
R="$MODULE/build-brick/host/recompiler"
if [ ! -x "$R/psxrecomp-game" ]; then
    $SBX cmake -S "$SDK/psxrecomp/recompiler" -B "$R" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPSXRECOMP_ENABLE_CHD=ON > build-brick/host-configure.log
fi
$SBX cmake --build "$R" -j"$JOBS" > build-brick/host-build.log

# the disc, staged from the CHD (cue + bins), and its boot EXE
CUE=$(sed -n 's/^disc = "\(.*\)"$/\1/p' game.toml)
if [ ! -f "$CUE" ]; then
    mkdir -p disc
    $SBX python3 "$SDK/psxrecomp/tools/psx_chd.py" "$CHD" --lib "$R/libchdr.so" --extract disc --cue-name "$(basename "$CUE")" > /dev/null
    $SBX python3 "$SDK/psxrecomp/tools/new_project_layout/probe_disc.py" "$CUE" --write-boot-exe disc --out-dir disc --disc-rel "$CUE" > /dev/null
fi

# the game's C
$SBX "$R/psxrecomp-game" --config "$MODULE/game.toml" > build-brick/generate.log

# the captured overlays, baked into native code (the bake checks the translator against a codegen hash a configure for this
# machine writes)
H="$MODULE/build-brick/host/module"
if [ -f overlays/overlay_captures.json ]; then
    $SBX cmake -S . -B "$H" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPSX_DEBUG_TOOLS=ON -DPSX_RECOMP_UI=OFF \
        -DPSX_NETPLAY=OFF -DPSX_REWIND=OFF -DPSX_ENABLE_VULKAN=OFF -DPSX_SDL3_FETCH=OFF -DPSX_ZLIB_FETCH=OFF \
        -DPSXRECOMP_BIOS_STEMS=OpenBIOS > build-brick/host-module-configure.log
    $SBX cmake --build "$H" --target psxrecomp_codegen_hash > /dev/null
    $SBX python3 "$SDK/psxrecomp/tools/compile_overlays.py" --static --captures overlays/overlay_captures.json \
        --game-toml game.toml --recompiler "$R/psxrecomp-game" --runtime-include "$SDK/psxrecomp/runtime/include" \
        --out-dir generated --gcc "$(command -v gcc)" --cps > build-brick/overlays.log
fi

# the module, for the handheld: PSXRecomp's SDL2 backend (the sysroot's), CHD discs, the presenter (brick/brick.cmake);
# pkg-config sees the sysroot's packages only, and no path of this machine goes into the binary
B="$MODULE/build-brick/build"
SYSROOT="$TC/aarch64-nextui-linux-gnu/libc"
mkdir -p build-brick/include
rm -rf build-brick/include/vulkan build-brick/include/vk_video
cp -r /usr/include/vulkan /usr/include/vk_video build-brick/include/
PKG_CONFIG_LIBDIR="$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig" PKG_CONFIG_SYSROOT_DIR="$SYSROOT" \
    $SBX cmake -S . -B "$B" -G Ninja \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -ffile-prefix-map=$MODULE=. -ffile-prefix-map=$SDK=sdk" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -ffile-prefix-map=$MODULE=. -ffile-prefix-map=$SDK=sdk" \
    -DCMAKE_TOOLCHAIN_FILE="$SDK/tools/toolchain/tg5040.cmake" -DTG5040_ROOT="$TC" -DCMAKE_BUILD_TYPE=Release \
    -DPSX_SDL_BACKEND=SDL2 -DPSX_RECOMP_UI=OFF -DPSX_NETPLAY=OFF -DPSX_REWIND=OFF -DPSX_ENABLE_VULKAN=OFF -DPSX_ZLIB_FETCH=OFF \
    -DPSX_DEBUG_TOOLS=OFF -DPSX_STATIC_RUNTIME=OFF -DPSXRECOMP_ENABLE_CHD=ON -DPSXRECOMP_BIOS_STEMS=OpenBIOS \
    -DDEADSTATION_VULKAN_INCLUDE="$MODULE/build-brick/include" > build-brick/build-configure.log
$SBX cmake --build "$B" --target psxrecomp_codegen_hash > /dev/null
$SBX cmake --build "$B" --target psx-runtime -j"$JOBS" "$@" > build-brick/build.log
BUILT=$(cat "$B/psxrecomp_exe_name-psx-runtime.txt")
[ "$BUILT" = "$EXEC" ] || { echo "build-module: the build made $BUILT, module.cfg says exec = $EXEC"; exit 2; }

# as Deadstation installs it
OUT="build-brick/$M"
rm -rf "$OUT"
mkdir -p "$OUT/bios" "$OUT/licenses"
llvm-strip -o "$OUT/$EXEC" "$B/$BUILT"
chmod +x "$OUT/$EXEC"
cp module.cfg store.json game.toml "$OUT/"
[ -f game_options.toml ] && cp game_options.toml "$OUT/"
# the pipelines it draws, built ahead of use from its first run on (brick/psx_gpu.cpp)
[ -f pipelines.keys ] && cp pipelines.keys "$OUT/"
# the mods the build staged (the module's own package, PSXRecomp's Fast Loading and CD Speed, ...)
[ -d "$B/mods/bundled" ] && mkdir -p "$OUT/mods" && cp -r "$B/mods/bundled" "$OUT/mods/"
cp "$SDK/psxrecomp/bios/openbios.bin" "$OUT/bios/"
cp "$SDK/psxrecomp/bios/OpenBIOS.LICENSE" "$OUT/licenses/OpenBIOS.LICENSE"
cp "$SDK/LICENSE" "$OUT/licenses/PolyForm-Noncommercial-1.0.0.txt"
cp "$SDK/THIRD_PARTY_NOTICES.md" "$OUT/licenses/THIRD_PARTY_NOTICES.md"
cp "$SDK/psxrecomp/THIRD_PARTY_ATTRIBUTION.md" "$OUT/licenses/PSXRecomp-THIRD_PARTY_ATTRIBUTION.md"
echo "build-module: $OUT ($(du -sh "$OUT" | cut -f1))"
