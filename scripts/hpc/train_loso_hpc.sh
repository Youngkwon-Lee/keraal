#!/bin/bash
#
# KERAAL LOSO Training Script for HPC
# Usage: bash scripts/hpc/train_loso_hpc.sh
#

echo "============================================================"
echo "KERAAL LOSO Training"
echo "Model: LSTM Classifier"
echo "Evaluation: Leave-One-Subject-Out"
echo "============================================================"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keraal

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Navigate to project
cd ~/keraal

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/loso_${TIMESTAMP}.log"

echo "Starting training..."
echo "Log file: ${LOG_FILE}"
echo ""

python train_loso.py 2>&1 | tee ${LOG_FILE}

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Results: ${LOG_FILE}"
echo "============================================================"
