#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name="active_learning"
#SBATCH --output=/users/lucelo/logs/slurm-%j.out
#SBATCH --error=/users/lucelo/logs/slurm-%j.err
#SBATCH --account=lucelo

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
--input_size 4096 \
--dataset_type "numpy" \
--clusters_filepath "/users/lucelo/UQLRM/uqlrm/data/groups_reddit.txt" \
--per_device_eval_batch_size 1024 \
--quantization_scheme "none" \
--push_predictions_to_hub True \
--predictions_dataset_hub "luckeciano/uqlrm_predictions" \
--use_peft False \
--undersample_eval False \
--undersample_val_ratio 1.0  \
--undersample_infer_ratio 1.0 \
--initial_sample_size 320 \
--ensemble_size 5 \
--active_batch_size 320 \
--epoch_steps 75 \
--per_device_train_batch_size 32 \
--save_predictions_steps 1 \
--gradient_accumulation_steps 1 \
--heuristic "Epistemic Uncertainty" \
--selection_strategy "batch-state-entropy" \
--normalize_state_features "True" \
--normalize_entropy "False" \
--no_uncertainty "False" \
--state_features_dataset_name "luckeciano/hermes-reddit-post-features" \
--state_ent_beta "1e-2" \
--state_ent_k "13" \
--gumbel_beta "8.0" \
--gumbel_beta_annealing "True" \
--gumbel_beta_annealing_epochs "50" \
--pool_size 92000 \
--seed 712 \
--score_init_std 0.02 \
--learning_rate 3e-5 \
--bf16 False \
--train_split "train" \
--test_split "test" \
--valid_split "eval" \
--ood_split "ood" \
--gradient_checkpointing False  \
--dataset_strategy "full_labeled_set" \
--training_strategy "full_retrain" \
--ignore_data_skip True \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--eval_steps 100 \
--save_steps 100 \
--regularization_loss "False" \
--lambda_regularizer "1.0" \
--log_batch_indices "True" \
--batch_idx_filepath "/users/lucelo/UQLRM/uqlrm/data/batch_ids_$1.csv" \
