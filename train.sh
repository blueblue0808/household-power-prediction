#!/bin/bash

MODEL="lstm"           # 可选: lstm, patch, transformer
TASK="short"           # 可选: short, long
SEED=42                # 随机种子

HIDDEN_DIM=128         # 隐藏层维度
NUM_LAYERS=2           # 层数
DROPOUT=0.2            # Dropout 比率

BATCH_SIZE=32          # 批次大小
LR=1e-3                # 学习率
EPOCHS=100             # 最大训练轮数

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