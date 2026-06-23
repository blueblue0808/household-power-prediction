#!/bin/bash

MODEL="patch"           # 可选: lstm, patch, transformer
TASK="long"           # 可选: short, long

HIDDEN_DIM=128
NUM_LAYERS=2
DROPOUT=0.2

BATCH_SIZE=32
LR=1e-3
EPOCHS=100

SEEDS=(42 43 44 45 46)  # 5个随机种子

for SEED in "${SEEDS[@]}"
do
    echo "=============================="
    echo "Model: ${MODEL}, Task: ${TASK}, Seed: ${SEED}"
    echo "=============================="

    python train.py \
        --model ${MODEL} \
        --task ${TASK} \
        --seed ${SEED} \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS} \
        --dropout ${DROPOUT} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --epochs ${EPOCHS}

    python evaluate.py \
        --model ${MODEL} \
        --task ${TASK} \
        --seed ${SEED} \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS} \
        --dropout ${DROPOUT}
done

python summary.py \
    --model ${MODEL} \
    --task ${TASK}