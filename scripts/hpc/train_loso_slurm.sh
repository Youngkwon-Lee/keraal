#!/bin/bash
#SBATCH --job-name=keraal_loso
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#
# KERAAL LOSO Training - SLURM Job Script
# Submit: sbatch scripts/hpc/train_loso_slurm.sh
#

echo "============================================================"
echo "KERAAL LOSO Training (SLURM Job)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

# Load modules (adjust for your HPC)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keraal

# Navigate to project
cd ~/keraal

# GPU info
echo ""
echo "GPU Status:"
nvidia-smi
echo ""

# Run training
echo "Starting LOSO training..."
python train_loso.py

echo ""
echo "============================================================"
echo "Job Complete!"
echo "============================================================"
