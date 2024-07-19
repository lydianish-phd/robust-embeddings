#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=8               # number of tasks per node
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:10:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile
source $HOME/.bash_python_exports

module load cpuarch/amd pytorch-gpu/py3/2.0.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

TEACHER_MODEL=$LASER/models/laser2.pt
STUDENT_MODEL=$MODELS/roberta-base/pytorch_model.bin
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_035
OUTPUT_DIR=$EXPERIMENT_DIR/models
TENSORBOARD_DIR=$EXPERIMENT_DIR/tensorboard
CONFIG_PATH=$EXPERIMENT_DIR/train-config.json


accelerate launch $HOME/robust-embeddings/src/sonar/train.py

echo "Done..."

