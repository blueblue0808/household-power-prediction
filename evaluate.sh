#!/bin/bash

# 必须与训练时一致

MODEL="lstm"         
TASK="short"          
SEED=42             

HIDDEN_DIM=128       
NUM_LAYERS=2       
DROPOUT=0.2       


python evaluate.py \
    --model ${MODEL} \
    --task ${TASK} \
    --seed ${SEED} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --dropout ${DROPOUT}