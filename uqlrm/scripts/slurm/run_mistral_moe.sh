#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="rw_infer_hermes"
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

python ~/UQLRM/uqlrm/run_mistral_moe.py


