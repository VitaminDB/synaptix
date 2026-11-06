#!/bin/sh
# Каталог для `synaptix convert` из снимка HF Supertone/supertonic-3:
#   stage_bundle.sh <снимок> <стадия>   (нужен собранный пример synaptix-io onnx_dump)
set -e
S="$1"; ST="$2"; D="$(dirname "$0")/../../../../../target/fast-release/examples/onnx_dump"
for m in duration_predictor text_encoder vector_estimator vocoder; do
  mkdir -p "$ST/$m"; "$D" "$S/onnx/$m.onnx" --weights "$ST/$m/model.safetensors"
done
cp "$S/onnx/tts.json" "$S/onnx/unicode_indexer.json" "$S/LICENSE" "$S/README.md" "$ST/"
# voices.json: {"F1": {...}, …} — все голоса одним файлом
{ printf '{'; sep=''; for f in "$S"/voice_styles/*.json; do n=$(basename "$f" .json); printf '%s"%s":' "$sep" "$n"; cat "$f"; sep=','; done; printf '}'; } > "$ST/voices.json"
echo "synaptix convert $ST supertonic-3.syn --arch supertonic --id supertonic-3"
