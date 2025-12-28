#!/bin/bash

# Check if tsp is installed
if ! command -v tsp &> /dev/null; then
    echo "Error: tsp (Task Spooler) is not installed."
    echo "Please install it using: sudo apt-get install task-spooler"
    echo "On some systems, the command might be 'ts'."
    exit 1
fi

# Set the number of concurrent jobs (slots)
# Default to 1 (sequential execution)
# If you have enough GPUs to run multiple jobs at once, increase this number.
# Note: Each job uses 2 GPUs by default (cuda:0 and cuda:1).
SLOTS=${1:-1}
tsp -S $SLOTS

echo "Queueing experiments to tsp (Slots: $SLOTS)..."

# ==========================================
# 1. GRPO Baseline
# ==========================================
# Default: lr=1e-5, loss_type=reinforce_with_baseline
echo "Queueing: grpo_baseline"
tsp ./scripts/run_experiment.sh grpo_baseline

echo "Queueing: masked_normalization"
tsp ./scripts/run_experiment.sh masked_normalization  --use_masked_normalization
# ==========================================
# 2. Learning Rate Ablations
# ==========================================
echo "Queueing: grpo_lr_5e-6"
tsp ./scripts/run_experiment.sh grpo_lr_5e-6 --learning-rate 5e-6

echo "Queueing: grpo_lr_2e-5"
tsp ./scripts/run_experiment.sh grpo_lr_2e-5 --learning-rate 2e-5

# ==========================================
# 3. Loss Type Ablations
# ==========================================
# echo "Queueing: grpo_clip_loss"
# tsp ./scripts/run_experiment.sh grpo_clip_loss --loss-type grpo_clip

echo "Queueing: grpo_no_baseline"
tsp ./scripts/run_experiment.sh grpo_no_baseline --loss-type no_baseline

echo "------------------------------------------------"
echo "All jobs queued!"
echo "Run 'tsp' to see the queue status."
echo "Run 'tsp -C' to clear finished jobs."
