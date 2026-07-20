#!/usr/bin/env bash
# T4 stack_bowls_three: final T3-code/T4-weight profile (3-GPU DDP by default).
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACT_TRACK=T4
export ACT_PROFILE="${ACT_PROFILE:-T4-h1024-c100-ff4096-ddp3}"
export ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-100}"
export ACT_HIDDEN_DIM="${ACT_HIDDEN_DIM:-1024}"
export ACT_DIM_FEEDFORWARD="${ACT_DIM_FEEDFORWARD:-4096}"
export ACT_NUM_EPOCHS="${ACT_NUM_EPOCHS:-8000}"
export ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-250}"
export ACT_VAL_FREQ="${ACT_VAL_FREQ:-50}"
export ACT_NPROC_PER_NODE="${ACT_NPROC_PER_NODE:-3}"
export ACT_AUG="${ACT_AUG:-0}"

seed=${1:-0}
gpu_ids=${2:-0,1,2}
exec bash "$script_dir/train.sh" stack_bowls_three stack_bowls_three_1021ep 1021 "$seed" "$gpu_ids" "${@:3}"
