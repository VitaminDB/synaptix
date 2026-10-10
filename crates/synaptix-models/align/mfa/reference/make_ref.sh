#!/usr/bin/env bash
# Эталоны Kaldi для гейта выравнивателя MFA.
#   reference/make_ref.sh <env MFA> <распакованная модель> <wav 16k> <out>
set -euo pipefail
ENV="$1"; M="$2"; WAV="$3"; OUT="$4"
mkdir -p "$OUT"; B="$ENV/bin"
echo "utt $WAV" > "$OUT/wav.scp"
"$B/compute-mfcc-feats" --dither=0 --snip-edges=false --num-mel-bins=23 --num-ceps=13 --low-freq=20 --high-freq=7800 \
  --use-energy=false --cepstral-lifter=22 --preemphasis-coefficient=0.97 --frame-shift=10 --frame-length=25 \
  scp:"$OUT/wav.scp" ark,t:"$OUT/mfcc.txt"
"$B/compute-cmvn-stats" ark,t:"$OUT/mfcc.txt" ark:"$OUT/cmvn.ark"
"$B/apply-cmvn" --norm-vars=false ark:"$OUT/cmvn.ark" ark,t:"$OUT/mfcc.txt" ark:- \
 | "$B/splice-feats" --left-context=3 --right-context=3 ark:- ark:- \
 | "$B/transform-feats" "$M/lda.mat" ark:- ark,t:"$OUT/feats.txt"
"$B/gmm-compute-likes" "$M/final.alimdl" ark,t:"$OUT/feats.txt" ark,t:"$OUT/likes.txt"
"$B/gmm-copy" --binary=false "$M/final.alimdl" "$OUT/final.alimdl.txt"
"$B/copy-tree" --binary=false "$M/tree" "$OUT/tree.txt"
"$B/copy-matrix" --binary=false "$M/lda.mat" "$OUT/lda.txt"
