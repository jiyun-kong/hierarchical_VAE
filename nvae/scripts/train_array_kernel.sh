#!/bin/bash
#SBATCH --job-name=nvae_exp
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

CONFIGS=("configs/exp5_kernel7.yaml")

for CONFIG in "${CONFIGS[@]}"
do
    echo "Running with $CONFIG"
    python train_kernel.py --config $CONFIG
done