#!/usr/bin/env bash
# T1 adjust_bottle: official small ACT baseline.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACT_TRACK=T1
export ACT_PROFILE="${ACT_PROFILE:-T1-h512-c50-ff3200}"
export ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-50}"
export ACT_HIDDEN_DIM="${ACT_HIDDEN_DIM:-512}"
export ACT_DIM_FEEDFORWARD="${ACT_DIM_FEEDFORWARD:-3200}"
export ACT_NUM_EPOCHS="${ACT_NUM_EPOCHS:-6000}"
export ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-2000}"
export ACT_VAL_FREQ="${ACT_VAL_FREQ:-50}"
export ACT_NPROC_PER_NODE="${ACT_NPROC_PER_NODE:-1}"
export ACT_AUG="${ACT_AUG:-0}"

seed=${1:-0}
gpu_ids=${2:-0}
exec bash "$script_dir/train.sh" adjust_bottle adjust_bottle_200ep 200 "$seed" "$gpu_ids" "${@:3}"
