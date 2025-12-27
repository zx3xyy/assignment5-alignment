#!/bin/bash
# Usage: ./scripts/run_experiment.sh [EXP_NAME] [EXTRA_ARGS...]

killall python
killall VLLM::EngineCore

EXP_NAME=${1:-"grpo_baseline"}
shift

# Create output directory if it doesn't exist
mkdir -p cs336_alignment/results

# Run the training script
# The script itself will handle logging to cs336_alignment/results/$EXP_NAME/train.log
# and saving config to cs336_alignment/results/$EXP_NAME/config.json

echo "Starting experiment: $EXP_NAME"
echo "Logs will be saved to: cs336_alignment/results/$EXP_NAME/train.log"

# Ensure the directory exists
mkdir -p "cs336_alignment/results/$EXP_NAME"

# Run with stderr redirection to the log file
python cs336_alignment/grpo_train.py \
    --exp-name "$EXP_NAME" \
    "$@" 2>> "cs336_alignment/results/$EXP_NAME/train.log"

