#!/bin/bash

#SBATCH --job-name=plot      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc
source $HOME/.bash_profile

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r
SCORES_DIR=$EXPERIMENT_DIR/scores

METRICS="bleu comet chrf2"

python $HOME/robust-embeddings/src/sonar/plot-proba.py \
    -d $SCORES_DIR \
    -m $METRICS \

echo "Done..."
