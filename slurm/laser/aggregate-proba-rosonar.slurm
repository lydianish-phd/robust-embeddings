#!/bin/bash

#SBATCH --job-name=aggregate      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_BASE_NUM="048"

ALL_MODELS="LASER RoLASER-0.1 RoLASER-0.2 RoLASER-0.3"
METRICS="xsimpp"
SEEDS="1005 1006 1007 1008 1009"
PROBAS="0.0 0.1 0.2 0.3 0.4 0.5"

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_eval
INPUT_DIR=$EXPERIMENT_DIR/scores
CORPUS=flores200
CORPUS_PARTS="dev"

SRC_PATH=$HOME/robust-embeddings/src/laser

python $SRC_PATH/aggregate_proba.py \
    -i $INPUT_DIR \
    -m ${ALL_MODELS[@]} \
    -c $CORPUS \
    --corpus-parts $CORPUS_PARTS \
    --metrics $METRICS \
    --seeds $SEEDS \
    --probas $PROBAS \

for CORPUS_PART in $CORPUS_PARTS; do
    for METRIC in $METRICS; do
        SCORES_FILE=$INPUT_DIR/${CORPUS}-${CORPUS_PART}-${METRIC}_matrix.csv  
        python $SRC_PATH/plot-proba.py \
            -i $SCORES_FILE \
            -m $METRIC
    done
done


echo "Done..."
