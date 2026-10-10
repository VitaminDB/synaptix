#!/usr/bin/env bash
# Модель MFA (zip) → .mfa для synaptix-align-mfa. Нужны утилиты Kaldi из
# окружения MFA (gmm-copy, copy-tree, copy-matrix) — только при конвертации.
#   reference/convert.sh <env MFA> <model.zip> <словарь.dict> <out.mfa> [alimdl|mdl]
set -euo pipefail
ENV="$1"; ZIP="$2"; DICT="$3"; OUT="$4"; WHICH="${5:-alimdl}"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
unzip -q "$ZIP" -d "$T"; M="$(dirname "$(find "$T" -name meta.json | head -1)")"
MDL="$M/final.$WHICH"; [ -f "$MDL" ] || MDL="$M/final.mdl"
"$ENV/bin/gmm-copy" --binary=false "$MDL" "$T/mdl.txt" 2>/dev/null
"$ENV/bin/copy-tree" --binary=false "$M/tree" "$T/tree.txt" 2>/dev/null
"$ENV/bin/copy-matrix" --binary=false "$M/lda.mat" "$T/lda.txt" 2>/dev/null
"$(dirname "$0")/../../../../../target/fast-release/examples/mfa_convert" "$T/mdl.txt" "$T/tree.txt" "$T/lda.txt" "$M/meta.json" "$DICT" "$OUT"
