#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
resume_ckpt=${6:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}

cmd=(
    python3 imitate_episodes.py
    --task_name "sim-${task_name}-${task_config}-${expert_data_num}"
    --ckpt_dir "./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}"
    --policy_class ACT
    --kl_weight 10
    --chunk_size 64
    --hidden_dim 512
    --batch_size 32
    --dim_feedforward 8192
    --num_epochs 8000
    --lr 1e-5
    --save_freq 250
    --state_dim 16
    --seed "${seed}"
)

if [[ -n "${resume_ckpt}" ]]; then
    cmd+=(--resume_ckpt "${resume_ckpt}")
fi

"${cmd[@]}"
