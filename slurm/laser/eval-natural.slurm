#!/bin/bash

#SBATCH --job-name=eval-natural      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

source $HOME/.bashrc
source $HOME/.bash_profile
source activate laser_env

#------------------ LEXNORM ------------------#

# NOISY_LANGS=cleaned.uncased.raw.src
# TARGET_LANG=cleaned.uncased.raw.ref
# EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_025_lrec
# INPUT_DIR=$EXPERIMENT_DIR/tok/laser/
# CORPUS=lexnorm2015
# CORPUS_PARTS="train test"

#------------------ ROCSMT ------------------#

NOISY_LANGS=cleaned.src.raw-manseg.en
TARGET_LANG=cleaned.src.norm-manseg.en
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_025_lrec
INPUT_DIR=$EXPERIMENT_DIR/tok/laser/
CORPUS=rocsmt
CORPUS_PARTS="test"

for CORPUS_PART in $CORPUS_PARTS
do
    OUTPUT_DIR=$INPUT_DIR/$CORPUS/$CORPUS_PART
    
    echo "$CORPUS_PART"
    
    echo " - calculating xsim and cos dist"

    python3 $LASER/source/eval.py                \
        --base-dir $INPUT_DIR                         \
        --corpus $CORPUS                        \
        --corpus-part $CORPUS_PART               \
        --margin ratio                           \
        --src-encoder   $LASER/models/laser2.pt  \
        --src-vocab-file $LASER/models/laser2.cvocab \
        --src-langs $NOISY_LANGS      \
        --tgt-langs $TARGET_LANG \
        --output-dir $OUTPUT_DIR \
        --cosine-distances \
        --verbose 

done

echo "Done..."