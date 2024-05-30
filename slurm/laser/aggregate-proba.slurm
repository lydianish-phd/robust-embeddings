#!/bin/bash

#SBATCH --job-name=aggregate      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=9

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

MODEL_NAME[0]=laser
MODEL_NAME[1]=roberta-maxpool-init-0.1
MODEL_NAME[2]=roberta-maxpool-init-0.2
MODEL_NAME[3]=roberta-maxpool-init-0.3
MODEL_NAME[4]=roberta-maxpool-init-0.4
MODEL_NAME[5]=roberta-maxpool-init-0.5
MODEL_NAME[6]=roberta-maxpool-init-0
MODEL_NAME[7]=roberta-maxpool-init-0.35
MODEL_NAME[8]=roberta-maxpool-init-0.45
MODEL_NAME[9]=roberta-maxpool-init-0.6

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_042_eval

INPUT_DIR=$EXPERIMENT_DIR/scores
CORPUS_PARTS="devtest"

python $HOME/robust-embeddings/src/laser/aggregate_paired.py -i $INPUT_DIR -m ${MODEL_NAME[$SLURM_ARRAY_TASK_ID]} -p $CORPUS_PARTS

echo "Done..."
