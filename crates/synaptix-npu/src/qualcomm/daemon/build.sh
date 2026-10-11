#!/bin/bash
set -euo pipefail
: "${QNN_SDK_ROOT:?QNN_SDK_ROOT — каталог QAIRT (include/QNN)}"
S=$(dirname "$(realpath "$0")")
FLAGS=(-std=c++17 -O2 -g -Wall -Wno-missing-field-initializers -Wno-missing-braces -I"$QNN_SDK_ROOT/include/QNN")
if [ "${1:-}" = --host ]; then
    OUT=${2:-$PWD/synnpu-qnn-host}
    clang++ "${FLAGS[@]}" -o "$OUT" "$S/synnpu-qnn.cpp" -ldl -lpthread
else
    NDK=${NDK:-/opt/android-ndk}
    OUT=${1:-$PWD/synnpu-qnn}
    "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++" \
        "${FLAGS[@]}" -static-libstdc++ -o "$OUT" "$S/synnpu-qnn.cpp" -ldl -llog
fi
echo "$OUT"
