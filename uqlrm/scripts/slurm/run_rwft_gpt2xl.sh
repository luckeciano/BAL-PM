#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="gpt2xl_rwft"
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

python ~/UQLRM/src/scripts/reward_model_training.py \
--output_dir /scratch/lucelo/sft/results/$1_$2 \
--run_name "$1_$2" \
--dataset_name "luckeciano/learning-to-summarize" \
--per_device_eval_batch_size 64 \
--model_name "luckeciano/merged-gpt2-xl-sft-reddit" \
--quantization_scheme "none" \
--push_predictions_to_hub True \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--use_peft False \
--eval_steps 10 \
--logging_steps 10 \
--save_steps 10 \
--num_train_epochs 1 \
--undersample_eval True \
--undersample_ratio 0.1 \
