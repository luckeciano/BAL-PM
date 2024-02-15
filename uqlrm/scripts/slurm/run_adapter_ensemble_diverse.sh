#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="vpo_mini"
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

SEED=$2
RANDOM=$SEED

layers=("[256]" "[2048, 256]" "[1024, 256]" "[512, 256]" "[2048, 512]" "[1024, 512]" "[512]" "[1024]" "[2048, 1024]" "[2048]")
act_fns=("relu" "sigmoid" "tanh" "leakyrelu")
w_init=(0.01 0.1 1.0)
init_fns=("normal" "xavier_normal" "uniform" "kaiming_normal" "xavier_uniform" "kaiming_uniform")
batch_size=(16 32 64 128 256 512 1024)
learning_rate=(0.001 0.0001 0.0003 0.00003 0.00001)

echo $1_$2

for (( i=1; i<=$3; i++ ))
do
  s=$RANDOM
  
  layer_idx=$((RANDOM % ${#layers[@]}))
  act_idx=$((RANDOM % ${#act_fns[@]}))
  winit_idx=$((RANDOM % ${#w_init[@]}))
  init_idx=$((RANDOM % ${#init_fns[@]}))
  batch_idx=$((RANDOM % ${#batch_size[@]}))
  lr_idx=$((RANDOM % ${#learning_rate[@]}))

  echo ${layers[$layer_idx]}
  echo ${act_fns[$act_idx]}
  echo ${w_init[$winit_idx]}
  echo ${init_fns[$init_idx]}
  echo ${learning_rate[$lr_idx]}
  echo ${batch_size[$batch_idx]}

  python ~/UQLRM/uqlrm/adapter_ensemble_reward_training.py \
    --output_dir /scratch-ssd/lucelo/mini_vpo/results/$1_$s \
    --run_name "$1_$" \
    --dataset_name luckeciano/reddit-features-hermes \
    --per_device_eval_batch_size 32 \
    --learning_rate ${learning_rate[$lr_idx]} \
    --per_device_train_batch_size ${batch_size[$batch_idx]} \
    --push_predictions_to_hub True \
    --save_predictions_steps 1 \
    --bf16 True \
    --layers "${layers[$layer_idx]}" \
    --activation_fn "${act_fns[$act_idx]}" \
    --weight_init ${w_init[$winit_idx]} \
    --init_func ${init_fns[$init_idx]} \
    --seed $s &
    sleep 20
done

sleep 28000