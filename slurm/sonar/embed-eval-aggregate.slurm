#!/bin/bash

#SBATCH --job-name=embed-eval-aggregate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=20	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:10:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047s

MODEL[0]=sonar
MODEL[1]=rosonar
MODEL[2]=rosonar_std
MODEL[3]=rolaser_v2

INPUT_DIR=$EXPERIMENT_DIR/scores

#------------------ FLORES AUGMENTED ------------------#

CORPUS=flores200/ugc
CORPUS_PARTS="devtest"
SEEDS="100 101 102 103 104 105 106 107 108 109"
METRICS="xsimpp"

echo "Aggregating $METRICS for $CORPUS $CORPUS_PARTS"

#loop through models
for i in {0..3}
do
    echo "- Model: ${MODEL[$i]}"
    python $HOME/robust-embeddings/src/laser/aggregate_paired.py \
        -i $INPUT_DIR \
        -m ${MODEL[$i]} \
        -p $CORPUS_PARTS \
        --metrics $METRICS \
        --seeds $SEEDS \
        --gold-model ${MODEL[0]}
done

#------------------ ROCSMT AUGMENTED ------------------#

CORPUS=rocsmt/artificial
CORPUS_PARTS="test"
SEEDS="1000 1001 1002 1003 1004"
PROBAS="0.05 0.1"
METRICS="xsim"

echo "Aggregating $METRICS for $CORPUS $CORPUS_PARTS"
echo "Models: ${MODEL[@]}"

python $HOME/robust-embeddings/src/laser/aggregate_proba.py \
    -i $INPUT_DIR \
    -m ${MODEL[@]} \
    -c $CORPUS \
    --corpus-parts $CORPUS_PARTS \
    --metrics $METRICS \
    --seeds $SEEDS \
    --probas $PROBAS \

echo "Done..."