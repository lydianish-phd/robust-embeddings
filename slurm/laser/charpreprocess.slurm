#!/bin/bash

#SBATCH --job-name=charpreprocess      # Job name
#SBATCH --account=ncm@cpu
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

set -e

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

TEACHER_VOCAB=$LASER/models/laser2.cvocab

TEACHER_LANG=laserstd
STUDENT_LANG[0]=charobertastd
STUDENT_LANG[1]=charobertaugc

DATA_DIR=$DATASETS/oscar/mini/4Mlrec

echo "Creating joint dictionary from standard and UGC data..."

TOK_DIR=$DATA_DIR/tok/${STUDENT_LANG[1]}-${STUDENT_LANG[0]}
BIN_DIR=$DATA_DIR/bin/${STUDENT_LANG[1]}-${STUDENT_LANG[0]}
STUDENT_PRETOK_VOCAB=$BIN_DIR/dict.${STUDENT_LANG[0]}.txt # joint std and ugc vocab

fairseq-preprocess --source-lang ${STUDENT_LANG[1]} --target-lang ${STUDENT_LANG[0]} \
    --trainpref $TOK_DIR/train \
    --destdir $BIN_DIR \
    --dataset-impl cached \
    --workers 16 \
    --joined-dictionary \
    --dict-only \

echo "Binarizing standard data..."

TOK_DIR=$DATA_DIR/tok/${STUDENT_LANG[0]}-$TEACHER_LANG
BIN_DIR=$DATA_DIR/bin/${STUDENT_LANG[0]}-$TEACHER_LANG

fairseq-preprocess --source-lang ${STUDENT_LANG[0]} --target-lang $TEACHER_LANG \
    --trainpref $TOK_DIR/train \
    --destdir $BIN_DIR \
    --srcdict $STUDENT_PRETOK_VOCAB \
    --tgtdict $TEACHER_VOCAB \
    --dataset-impl cached \
    --workers 16 \

echo "Binarizing UGC data..."

TOK_DIR=$DATA_DIR/tok/${STUDENT_LANG[1]}-$TEACHER_LANG
BIN_DIR=$DATA_DIR/bin/${STUDENT_LANG[1]}-$TEACHER_LANG

fairseq-preprocess --source-lang ${STUDENT_LANG[1]} --target-lang $TEACHER_LANG \
    --trainpref $TOK_DIR/train \
    --destdir $BIN_DIR \
    --srcdict $STUDENT_PRETOK_VOCAB \
    --tgtdict $TEACHER_VOCAB \
    --dataset-impl cached \
    --workers 16 \

echo "Done..."
