#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lucelo/logs/slurm-%j.out
#SBATCH --error=/home/lucelo/logs/slurm-%j.err

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/


/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/UQLRM/environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate uqrm


srun python ~/UQLRM/src/scripts/sft.py \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1