#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-__}"
DEVICE="${DEVICE:-cuda}"
EXP_ID="${EXP_ID:-0}"
DATASET="${DATASET:-norway_test}"
LAMBDA_1="${LAMBDA_1:-1.0}"
LAMBDA_2="${LAMBDA_2:-10.0}"

ONLY_COLOR=--save-only-color
if [[ -n "${DEBUG+x}" ]]; then
    ONLY_COLOR=
fi

SAVE_DIR=/images/colorized
MODE=test
DATA=lenticular_full_image
MODEL=UResNet
MODEL_2=UResNetColorize
RESUME_2=/checkpoints/model_colorize.pth.tar
STYLE=new_dolce

RESUME="/checkpoints/lenticular_square_patch/${MODEL}_${TAG}/experiment_${EXP_ID}/model_best.pth.tar"
DATAFOLDER="/images/raw/${DATASET}"


python -u colorize_suite.py \
    --tag "$TAG" \
    --device "$DEVICE" \
    --save-dir "$SAVE_DIR" \
    --mode "$MODE" \
    --data "$DATA" \
    --datafolder "$DATAFOLDER" \
    --model "$MODEL" \
    --model_2 "$MODEL_2" \
    --resume "$RESUME" \
    --resume_2 "$RESUME_2" \
    --style "$STYLE" \
    --lambda1 "$LAMBDA_1" \
    --lambda2 "$LAMBDA_2" \
    $ONLY_COLOR 



