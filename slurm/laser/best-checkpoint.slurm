#!/bin/bash

#SBATCH --job-name=best      # Job name
#SBATCH --account=rnh@cpu
##SBATCH --gres=gpu:1
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logiical
#SBATCH --time=0:10:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=110

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_048
INPUT_DIR=${EXPERIMENT_DIR}_valid/$SLURM_ARRAY_TASK_ID/scores
SRC_DIR=$HOME/robust-embeddings/src/laser

echo "Selecting best checkpoint..."

python $SRC_DIR/best.py -i $INPUT_DIR

echo "Renaming best checkpoint..."

python $SRC_DIR/rename_best.py -d $EXPERIMENT_DIR -s $SLURM_ARRAY_TASK_ID

python $SRC_DIR/plot_train_valid.py -i $INPUT_DIR #-y 0.005

echo "Done..."
