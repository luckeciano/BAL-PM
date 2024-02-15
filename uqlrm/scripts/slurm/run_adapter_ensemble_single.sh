#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name="vpo_mini"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/

export TMPDIR=/scratch/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/UQLRM/environment_cloud.yml
source /scratch-ssd/oatml/miniconda3/bin/activate uqrm

echo $TMPDIR

nvidia-smi

huggingface-cli login --token $HUGGINGFACE_WRITETOKEN


echo $1_$2

python ~/UQLRM/uqlrm/adapter_ensemble_reward_training.py \
  --output_dir /scratch-ssd/lucelo/adapter_ensemble_single/results/$1_$2 \
  --run_name "$1_$2" \
  --dataset_name luckeciano/reddit-features-hermes \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_epochs 10 \
  --push_predictions_to_hub True \
  --save_prediction_steps 1 \ 
  --bf16 False \
  --seed $2 



