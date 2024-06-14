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
--model_type "finetune_ens" \
--trainer_type "finetune_ensemble_trainer" \
--clusters_filepath "/users/lucelo/UQLRM/uqlrm/data/groups_reddit.txt" \
--dataset_strategy "full_labeled_set" \
--training_strategy "full_retrain" \
--collator_type "reward_collator" \
--output_dir /scratch-ssd/lucelo/active_learning/results/$1 \
--run_name "$1" \
--dataset_name "luckeciano/learning-to-summarize" \
--per_device_eval_batch_size 64 \
--model_name "gpt2-medium" \
--tokenizer_name "gpt2-medium" \
--quantization_scheme "none" \
--push_predictions_to_hub True \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--use_peft False \
--peft_lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
--undersample_eval True \
--undersample_val_ratio 0.01  \
--undersample_infer_ratio 0.1 \
--initial_sample_size 320 \
--ensemble_size 5 \
--active_batch_size 320 \
--per_device_train_batch_size 64 \
--save_predictions_steps 1 \
--gradient_accumulation_steps 1 \
--heuristic "random" \
--selection_strategy "sample-then-rank" \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--eval_steps "100" \
--save_steps "100" \
--ignore_data_skip "True" \
--pool_size 92000 \
--downsample_pool "True" \
--seed 88 \
--score_init_std 0.02 \
--learning_rate 1.41e-4 \
--bf16 True \
--gradient_checkpointing "True" \
--gumbel_beta "0.5" \
