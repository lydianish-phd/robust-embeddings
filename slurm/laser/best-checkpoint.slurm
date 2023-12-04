#!/bin/bash

#SBATCH --job-name=best      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile

INPUT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_028_valid/scores
SRC_DIR=$HOME/robust-embeddings/src/laser

echo "Selecting best checkpoint..."

python $SRC_DIR/best.py -i $INPUT_DIR

python $SRC_DIR/plot_train_valid.py -i $INPUT_DIR

echo "Done..."
