#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="active_learning"
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

huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

echo $1

python ~/UQLRM/uqlrm/active_learning.py \
--output_dir /scratch-ssd/lucelo/active_learning/results/$1 \
--model_type "adapters_ens" \
--run_name "$1" \
--dataset_name "luckeciano/reddit-features-hermes" \
--per_device_eval_batch_size 1024 \
--quantization_scheme "none" \
--push_predictions_to_hub True \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--use_peft False \
--undersample_eval False \
--undersample_ratio 0.1  \
--initial_sample_size 320 \
--ensemble_size 30 \
--active_batch_size 320 \
--per_device_train_batch_size 32 \
--save_predictions_steps 1 \
--gradient_accumulation_steps 1 \
--heuristic "random" \
--selection_strategy "rank" \
--pool_size 4096 \
--seed 66 \
--score_init_std 0.02 \
--learning_rate 3e-5 \
--bf16 True \
--train_split "train" \
--test_split "test" \
--valid_split "eval" \
--ood_split "ood" \
--gradient_checkpointing "False" \