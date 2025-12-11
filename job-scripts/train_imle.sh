#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=imle-train
#SBATCH --partition=gpubase_interac
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --output=/scratch/hsd31/%x_%j.out
#SBATCH --mail-user=hsd31@sfu.ca
#SBATCH --mail-type=END,FAIL

module load StdEnv/2023 python/3.11.5 scipy-stack cuda
source ~/py-311/bin/activate
export WANDB_API_KEY=""
cd ~/projects/aip-keli/hsd31/imle_policy_980
python -m torch.distributed.run --nproc_per_node=2 imle_policy/train.py --task zarr --method rs_imle
