#!/usr/bin/env bash
# Shared ACT training launcher. Prefer the named train_T1.sh ... train_T4.sh profiles.
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: bash train.sh <task> <task_config> <episodes> <seed> <gpu_ids> [extra imitate_episodes.py args...]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

task_name=$1
task_config=$2
expert_data_num=$3
seed=$4
gpu_ids=$5
shift 5
extra_args=("$@")

track=${ACT_TRACK:-custom}
chunk_size=${ACT_CHUNK_SIZE:-50}
hidden_dim=${ACT_HIDDEN_DIM:-512}
dim_feedforward=${ACT_DIM_FEEDFORWARD:-3200}
batch_size=${ACT_BATCH_SIZE:-8}
num_epochs=${ACT_NUM_EPOCHS:-6000}
save_freq=${ACT_SAVE_FREQ:-2000}
val_freq=${ACT_VAL_FREQ:-50}
nproc_per_node=${ACT_NPROC_PER_NODE:-1}
augmentation=${ACT_AUG:-0}
profile=${ACT_PROFILE:-${track}-h${hidden_dim}-c${chunk_size}-ff${dim_feedforward}}
ckpt_dir=${ACT_CKPT_DIR:-./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}-${profile}}

if ! [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACT_NPROC_PER_NODE must be a positive integer: $nproc_per_node" >&2
    exit 2
fi

IFS=',' read -r -a visible_gpus <<< "$gpu_ids"
if (( nproc_per_node > ${#visible_gpus[@]} )); then
    echo "Requested $nproc_per_node processes but CUDA_VISIBLE_DEVICES contains only $gpu_ids" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export ACT_AUG="$augmentation"

train_args=(
    --task_name "sim-${task_name}-${task_config}-${expert_data_num}"
    --ckpt_dir "$ckpt_dir"
    --policy_class ACT
    --kl_weight 10
    --chunk_size "$chunk_size"
    --hidden_dim "$hidden_dim"
    --batch_size "$batch_size"
    --dim_feedforward "$dim_feedforward"
    --num_epochs "$num_epochs"
    --lr 1e-5
    --save_freq "$save_freq"
    --val_freq "$val_freq"
    --state_dim 16
    --seed "$seed"
)
train_args+=("${extra_args[@]}")

if (( nproc_per_node > 1 )); then
    command=(torchrun --standalone --nproc_per_node="$nproc_per_node" imitate_episodes.py "${train_args[@]}")
else
    command=(python -u imitate_episodes.py "${train_args[@]}")
fi

echo "[train] track=$track task=$task_name episodes=$expert_data_num GPUs=$gpu_ids nproc=$nproc_per_node"
echo "[train] model: chunk=$chunk_size hidden=$hidden_dim ff=$dim_feedforward epochs=$num_epochs"
echo "[train] output: $ckpt_dir"
printf '[train] command:'
printf ' %q' "${command[@]}"
printf '\n'

if [ "${ACT_DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

exec "${command[@]}"
