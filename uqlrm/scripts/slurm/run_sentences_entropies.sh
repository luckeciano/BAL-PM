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

echo $1_$2

python ~/UQLRM/uqlrm/generate_sentence_entropies.py \
--output_dir /scratch-ssd/lucelo/rw_infer/results/$1_$2 \
--run_name "$1_$2" \
--dataset_name "luckeciano/learning-to-summarize" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--model_name "teknium/OpenHermes-2.5-Mistral-7B" \
--tokenizer_name "teknium/OpenHermes-2.5-Mistral-7B" \
--seed 42 \
--bf16 "True" \
# --seq_length 2048 \
# --train_split "train" \
# --test_split "test" \
# --valid_split "valid" \
# --ood_split "test" \


