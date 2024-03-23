#!/bin/bash

SLURM_ARRAY_TASK_ID=110

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

#module purge
#module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

#source $HOME/.bashrc
#source $HOME/.bash_profile
#source $HOME/.bash_python_exports

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_041
INPUT_DIR=${EXPERIMENT_DIR}_valid/$SLURM_ARRAY_TASK_ID/scores
SRC_DIR=$HOME/robust-embeddings/src/laser

echo "Selecting best checkpoint..."

python $SRC_DIR/best.py -i $INPUT_DIR

python $SRC_DIR/plot_train_valid.py -i $INPUT_DIR #-y 0.005

python $SRC_DIR/rename_best.py -d $EXPERIMENT_DIR -s $SLURM_ARRAY_TASK_ID

echo "Done..."
