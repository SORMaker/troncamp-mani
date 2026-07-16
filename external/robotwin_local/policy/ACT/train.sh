#!/bin/bash
set -euo pipefail

task_name=${1:-stack_bowls_two}
task_config=${2:-stack_bowls_two_400ep}
expert_data_num=${3:-400}
seed=${4:-0}
gpu_id=${5:-3}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export ACT_AUG=0

python -u imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}-big768-chunk100 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 768 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 500 \
    --val_freq 50 \
    --state_dim 16 \
    --seed ${seed}
