#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="sft_hermes"
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

python ~/UQLRM/uqlrm/sft.py \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--output_dir /scratch/lucelo/sft/results/hermes_sft \
--dataset_name "luckeciano/learning-to-summarize" \
--max_steps 32000 \
--model_name teknium/OpenHermes-2.5-Mistral-7B \
--peft_lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
--quantization_scheme "none" \
--use_peft True \
--peft_lora_r 8 \
--peft_lora_alpha 16 \
--peft_lora_dropout 0.05 \
--run_name "hermes-sft" \
--undersample_eval True \
--undersample_ratio 0.1 \
--learning_rate 3e-4 \
--no_model_cache True \
--eval_steps 500 \

