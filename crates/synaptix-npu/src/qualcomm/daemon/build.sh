#!/bin/bash
set -euo pipefail
: "${QNN_SDK_ROOT:?QNN_SDK_ROOT — каталог QAIRT (include/QNN)}"
NDK=${NDK:-/opt/android-ndk}
OUT=${1:-$PWD/synnpu-qnn}
S=$(dirname "$(realpath "$0")")
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++" \
    -std=c++17 -O2 -g -Wall -Wno-missing-field-initializers -Wno-missing-braces -I"$QNN_SDK_ROOT/include/QNN" \
    -static-libstdc++ -o "$OUT" "$S/synnpu-qnn.cpp" -ldl -llog
echo "$OUT"
