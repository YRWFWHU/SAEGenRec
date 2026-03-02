#!/bin/bash
# Launch SFT training with accelerate for multi-GPU support.
# Usage: ./scripts/run_sft_training.sh [config_path] [extra_args...]
CONFIG=${1:-configs/default.yaml}
shift 2>/dev/null
accelerate launch -m saegenrec.dataset train-sft "$CONFIG" "$@"
