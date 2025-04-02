#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r
MODEL="lai"

# Autoencode

DATA_DIR=$DATASETS/rocsmt/test
SRC_FILE=${DATA_DIR}/raw.en.test
REF_FILE=${DATA_DIR}/norm.en.test
SYS_FILE_SUFFIX=rocsmt/eng_Latn-eng_Latn/raw.en.test.out

SPM_MODEL=$MODELS/nllb600m/flores200sacrebleuspm

OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL
mkdir -p $OUTPUT_DIR/rocsmt/eng_Latn-eng_Latn/

echo "Copying $SRC_FILE to $OUTPUT_DIR/$SYS_FILE_SUFFIX"

cp $SRC_FILE $OUTPUT_DIR/$SYS_FILE_SUFFIX

SYS_FILES=("$OUTPUT_DIR/$SYS_FILE_SUFFIX")

echo "Evaluating $MODEL baseline on $SRC_FILE"

python $HOME/robust-embeddings/src/sonar/evaluate.py \
    --src-file ${SRC_FILE} \
    --sys-files "${SYS_FILES[@]}" \
    --ref-file ${REF_FILE}

# python $HOME/robust-embeddings/src/sonar/stats.py \
#     -i "${SYS_FILES[@]}" \
#     -m $SPM_MODEL

echo "Done..."

