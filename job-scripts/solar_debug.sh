#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=imle-train
#SBATCH --partition=mars-lab-short
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/%N-%j_compare_checkpoints.out
#SBATCH --error=logs/%N-%j_compare_checkpoints.err
#SBATCH --mail-user=dre3@sfu.ca
#SBATCH --mail-type=END,FAIL

echo "We did it!"