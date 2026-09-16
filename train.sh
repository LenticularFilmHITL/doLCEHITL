#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-__}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-norway_test}"
VAL_EPOCHS="${VAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_VAL="${TRAIN_VAL:-0.8}"
EPOCHS="${EPOCHS:-125}"
LR="${LR:-1e-05}"
LOGSTEP="${LOGSTEP:-25}"

SAVE_DIR=/checkpoints
MODE=train
DATA=lenticular_square_patch
MODEL=UResNet
RESUME=/checkpoints/model_lenticule_detection.pth.tar
OPTIMIZER=adam
SAVE_MODEL=best

DATAFOLDER="/datasets/${DATASET}"


python -u developing_suite.py \
    --tag "$TAG" \
    --device "$DEVICE" \
    --save-dir "$SAVE_DIR" \
    --mode "$MODE" \
    --data "$DATA" \
    --datafolder "$DATAFOLDER" \
    --model "$MODEL" \
    --resume "$RESUME" \
    --batch-size "$BATCH_SIZE" \
    --train-val-ratio "$TRAIN_VAL" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --logstep-train "$LOGSTEP" \
    --save-model "$SAVE_MODEL" \
    --val-every-n-epochs "$VAL_EPOCHS" \
    --optimizer "$OPTIMIZER"



