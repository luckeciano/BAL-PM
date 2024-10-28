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

# For llama-3 inference
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# pip install accelerate==0.29.2 datasets==2.16.1 evaluate==0.4.1 huggingface-hub==0.20.3 tokenizers==0.15.2 transformers==4.39.3
# pip install peft==0.5.0 bitsandbytes==0.41.1 trl==0.8.1

echo $TMPDIR

nvidia-smi

huggingface-cli login --token $HUGGINGFACE_WRITETOKEN

echo $1_$2

python ~/UQLRM/uqlrm/reward_model_inference.py \
--output_dir /scratch-ssd/lucelo/rw_infer/results/$1_$2 \
--run_name "$1_$2" \
--dataset_name "luckeciano/learning-to-summarize" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--model_name "teknium/OpenHermes-2.5-Mistral-7B" \
--tokenizer_name "teknium/OpenHermes-2.5-Mistral-7B" \
--inference True \
--push_predictions_to_hub False \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--undersample_eval False \
--undersample_ratio 0.001 \
--seed 42 \
--bf16 "True" \
--preprocess_fn "reddit_post_preprocess_fn" \


