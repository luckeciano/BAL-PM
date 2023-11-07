#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name="reward_ft"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/

export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p $TMPDIR
BUILD_DIR=/scratch-ssd/${USER}/conda_envs/pip-build

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ~/UQLRM/environment_cloud.yml
source /scratch-ssd/oatml/miniconda3/bin/activate uqrm

echo $TMPDIR

nvidia-smi

huggingface-cli login --token $HUGGINGFACE_TOKEN

python ~/UQLRM/src/scripts/reward_model_training.py \
--output_dir /scratch/lucelo/sft/results/gpt2_rwft_nopeft_loadcpkt_reddit_2 \
--run_name "gpt2_rwft_nopeft_loadckpt_reddit_2" \
--dataset_name "luckeciano/reddit-human-preferences" \
--model_name "luckeciano/gpt2-sft-reddit" \
--quantization_scheme "none" \
--use_peft False
