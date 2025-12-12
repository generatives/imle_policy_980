#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --job-name=imle-train
#SBATCH --partition=mars-lab-long
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --output=/project/mars-lab/dre3/projects/imle_policy_980/logs/%j_compare_checkpoints-%N.out
#SBATCH --error=/project/mars-lab/dre3/projects/imle_policy_980/logs/%j_compare_checkpoints-%N.err
#SBATCH --mail-user=dre3@sfu.ca
#SBATCH --mail-type=END,FAIL

LOGDIR=/project/mars-lab/dre3/projects/imle_policy_980/logs

configs=(
  zarr_config_full_data_3_large_model.json
  zarr_config_full_data_3_medium_model.json
  zarr_config_full_data_3_small_model.json
  zarr_config_full_data_6_medium_model.json
  zarr_config_full_data_9_medium_model.json
)

for cfg in "${configs[@]}"; do
  base="${cfg%.json}"
  srun --ntasks=1 --cpus-per-task=8 --gpus-per-task=2 \
       --output="${LOGDIR}/%j_${base}-%N.out" \
       --error="${LOGDIR}/%j_${base}-%N.err" \
       job-scripts/train_imle_solar.sh "$cfg" &
done

wait