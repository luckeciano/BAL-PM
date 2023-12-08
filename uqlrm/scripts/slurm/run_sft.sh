#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="sft"
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

python ~/UQLRM/src/scripts/sft.py \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--output_dir /scratch/lucelo/sft/results/gpt2_xl_reddit_lora_batch4 \
--model_name "gpt2-xl" \
--dataset_name "webis/tldr-17" \
--dataset_text_field "normalizedBody" \
--run_name "gpt2_xl_reddit_lora_batch4" \
--quantization_scheme "none" \
--max_steps 32000 \
--peft_lora_target_modules "c_attn" \
--peft_lora_r 16 \
--peft_lora_alpha 32 \
--test_split_size 0.0005 \
#--use_peft False \
#--num_train_epochs 3 \
#--peft_lora_target_modules "c_attn" \
#--peft_lora_r 16 \
#--peft_lora_alpha 32 \
