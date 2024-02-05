#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="hermes_rwft"
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

python ~/UQLRM/uqlrm/reward_model_training.py \
--output_dir /scratch-ssd/lucelo/sft/results/$1_$2 \
--run_name "$1_$2" \
--dataset_name "luckeciano/learning-to-summarize" \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--model_name "teknium/OpenHermes-2.5-Mistral-7B" \
--tokenizer_name "teknium/OpenHermes-2.5-Mistral-7B" \
--quantization_scheme "none" \
--push_predictions_to_hub True \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--use_peft True \
--peft_lora_r 2048 \
--peft_lora_alpha 4096 \
--eval_steps 5000 \
--logging_steps 5000 \
--save_steps 5000 \
--save_predictions_steps 5000 \
--num_train_epochs 10 \
--undersample_eval True \
--undersample_ratio 0.1 \
--learning_rate 3e-4 \
--num_warmup_steps 1000 \
--lr_scheduler_type cosine \
--peft_lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
