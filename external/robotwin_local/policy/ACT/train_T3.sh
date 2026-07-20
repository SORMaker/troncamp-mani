#!/usr/bin/env bash
# T3 stack_bowls_two: validated larger ACT profile.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACT_TRACK=T3
export ACT_PROFILE="${ACT_PROFILE:-T3-h768-c100-ff3200}"
export ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-100}"
export ACT_HIDDEN_DIM="${ACT_HIDDEN_DIM:-768}"
export ACT_DIM_FEEDFORWARD="${ACT_DIM_FEEDFORWARD:-3200}"
export ACT_NUM_EPOCHS="${ACT_NUM_EPOCHS:-6000}"
export ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-500}"
export ACT_VAL_FREQ="${ACT_VAL_FREQ:-50}"
export ACT_NPROC_PER_NODE="${ACT_NPROC_PER_NODE:-1}"
export ACT_AUG="${ACT_AUG:-0}"

seed=${1:-0}
gpu_ids=${2:-0}
exec bash "$script_dir/train.sh" stack_bowls_two stack_bowls_two_400ep 400 "$seed" "$gpu_ids" "${@:3}"
