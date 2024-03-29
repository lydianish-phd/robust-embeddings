#!/bin/bash

#SBATCH --job-name=validate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=6:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1-5

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_BASE_NUM="042"

MODEL_NAME[0]=laser
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py

EXPERIMENTS_DIR=$EXPERIMENTS/robust-embeddings/laser
DATASETS_DIR=$DATASETS/oscar/mini

MODEL_NAME[1]=roberta-maxpool-0.1
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}/models
VOCAB_FILE[1]=$DATASETS_DIR/4M_10/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=roberta-maxpool-0.2
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}b/models
VOCAB_FILE[2]=$DATASETS_DIR/4M_20/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=roberta-maxpool-0.3
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}c/models
VOCAB_FILE[3]=$DATASETS_DIR/4M_30/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[4]=roberta-maxpool-0.4
MODEL_PATH[4]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}d/models
VOCAB_FILE[4]=$DATASETS_DIR/4M_40/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[4]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[5]=roberta-maxpool-0.5
MODEL_PATH[5]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}e/models
VOCAB_FILE[5]=$DATASETS_DIR/4M_50/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[5]=$HOME/data-preparation/src/char-tokenizer.py

VALIDATION_SEED=110
OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}_valid
PREFIX=$DATASETS/flores200/ugc/$VALIDATION_SEED/dev

python $HOME/robust-embeddings/src/laser/validate.py \
    --teacher-model ${MODEL_NAME[0]} \
    --teacher-model-path ${MODEL_PATH[0]} \
    --teacher-vocab ${VOCAB_FILE[0]} \
    --teacher-tok ${TOKENIZER[0]} \
    --student-model ${MODEL_NAME[$SLURM_ARRAY_TASK_ID]} \
    --student-model-dir ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]} \
    --student-vocab ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
    --student-tok ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
    --ugc-file $PREFIX/eng_Latn_mix_all.dev \
    --std-file $PREFIX/eng_Latn.dev \
    --output-dir $OUTPUT_EXPERIMENT_DIR \

echo "Done..."
