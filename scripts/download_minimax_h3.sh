#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINIMAX_H3_ROOT:-$HOME/.local/share/synthos/hf/MiniMax-H3}"
VARIANT="${MINIMAX_H3_VARIANT:-FL2VA}"
REPO="MiniMaxAI/MiniMax-H3"

mkdir -p "$ROOT" "$ROOT/lora"

log() { printf '\n=== %s ===\n' "$*"; }

free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

need() {
    local want=$1 have
    have=$(free_gb)
    if [ "$have" -lt "$want" ]; then
        echo "недостаточно места: нужно ${want}G, свободно ${have}G" >&2
        exit 1
    fi
}

fetch() {
    local pattern=$1
    hf download "$REPO" --include "$pattern" --local-dir "$ROOT"
}

log "конфиги, токенайзер, процессор"
need 1
fetch "model_index.json"
fetch "$VARIANT/model_index.json"
fetch "$VARIANT/tokenizer/*"
fetch "$VARIANT/processor/*"
fetch "$VARIANT/transformer/config.json"
fetch "$VARIANT/transformer/model.safetensors.index.json"
fetch "scheduler/*"
fetch "audio_scheduler/*"

log "audio VAE (0.6 ГБ)"
need 2
fetch "$VARIANT/audio_vae/*"

log "video VAE (10.4 ГБ)"
need 12
fetch "$VARIANT/video_vae/**"

log "Turbo LoRA (744 МБ)"
need 2
hf download larryvrh/MiniMax-H3-Turbo-Lora \
    --include "minimax_h3_turbo_v4_step600_ema.safetensors" \
    --local-dir "$ROOT/lora/turbo"

log "Prompt Rewriter LoRA"
need 3
hf download lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA --local-dir "$ROOT/lora/prompt-rewriter"

log "DiT transformer, 13 шардов (62.5 ГБ)"
need 65
fetch "$VARIANT/transformer/*"

log "text encoder Qwen3-VL-32B, 14 шардов (67.8 ГБ)"
need 70
fetch "$VARIANT/text_encoder/*"

log "готово"
du -sh "$ROOT"
df -h "$ROOT" | tail -1
