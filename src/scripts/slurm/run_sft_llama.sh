#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="sft_llama"
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
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--output_dir /scratch/lucelo/sft/results/llama_lora_32ksteps_noquant_batch3_test \
--max_steps 32000 \
--model_name meta-llama/Llama-2-7b-hf \
--quantization_scheme "none" \
--peft_lora_target_modules "q_proj" "v_proj" \
--peft_lora_r 8 \
--peft_lora_alpha 16 \
--peft_lora_dropout 0.05 \
--run_name "llama-sft_lora_32ksteps_noquant_test" \
--test_split_size 0.0005 \
--no_model_cache True

