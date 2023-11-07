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

python ~/UQLRM/src/scripts/sft.py \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--output_dir /scratch/lucelo/sft/results/gpt2_nopeft_batch8_5epochs_a100 \
--run_name "gpt2_nopeft_batch8_5epochs_a100" \
--use_peft False \
--quantization_scheme "none" \
--num_train_epochs 5 \
#--peft_lora_target_modules "c_attn" \
#--peft_lora_r 16 \
#--peft_lora_alpha 32 \
