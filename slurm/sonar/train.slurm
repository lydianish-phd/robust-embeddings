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
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

#module load anaconda-py3/2023.09 pytorch-gpu/py3/2.3.0
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

TEACHER_MODEL=$LASER/models/laser2.pt
STUDENT_MODEL=$MODELS/roberta-base/pytorch_model.bin
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
CONFIG_PATH=$EXPERIMENT_DIR/train-config.json
SEED=42

accelerate launch $HOME/robust-embeddings/src/sonar/train.py -o $EXPERIMENT_DIR --seed $SEED 

echo "Done..."

