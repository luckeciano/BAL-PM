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



echo $1_$2

for (( i=1; i<=$3; i++ ))
do
  s=$RANDOM
  python ~/UQLRM/uqlrm/adapter_ensemble_reward_training.py \
    --output_dir /scratch-ssd/lucelo/mini_vpo/results/$1_$s \
    --run_name "$1_$s" \
    --dataset_name luckeciano/reddit-features-hermes \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --push_predictions_to_hub True \
    --bf16 True \
    --weight_init 0.1 \
    --seed $s & > /dev/null 2>&1 &
    sleep 40
done

sleep 18000