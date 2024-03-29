#!/bin/bash

#SBATCH --job-name=preprocess      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=10,20,30,40,50

echo "### Running $SLURM_JOB_NAME ###"

set -e

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

BACKBONE_MODEL=roberta-base
STUDENT_VOCAB=$MODELS/$BACKBONE_MODEL/$BACKBONE_MODEL.cvocab
TEACHER_VOCAB=$LASER/models/laser2.cvocab
TEACHER_LANG=laserstd
STUDENT_LANG[0]=robertastd
STUDENT_LANG[1]=robertaugc

DATA_PARTITION=1

DATA_DIR=$DATASETS/oscar/mini/4M_$SLURM_ARRAY_TASK_ID
TOK_DIR=$DATA_DIR/tok/${STUDENT_LANG[$DATA_PARTITION]}-$TEACHER_LANG
BIN_DIR=$DATA_DIR/bin/${STUDENT_LANG[$DATA_PARTITION]}-$TEACHER_LANG

echo "Creating dictionary and binarizing data..."

fairseq-preprocess --source-lang ${STUDENT_LANG[$DATA_PARTITION]} --target-lang $TEACHER_LANG \
    --trainpref $TOK_DIR/train \
    --destdir $BIN_DIR \
    --tgtdict $TEACHER_VOCAB \
    --srcdict $STUDENT_VOCAB \
    --dataset-impl cached \
    --workers 16 \

echo "Done..."
