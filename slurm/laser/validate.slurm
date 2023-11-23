#!/bin/bash

#SBATCH --job-name=validate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1-4

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile

# to be able to use fairseq files
FAIRSEQ_PATH=$HOME/fairseq/
export PYTHONPATH=$PYTHONPATH:$FAIRSEQ_PATH

# to be able to use LASER files
LASER_PATH=$HOME/LASER/source/
export PYTHONPATH=$PYTHONPATH:$LASER_PATH

MODEL_NAME[0]=laser
MODEL_PATH[0]=$MODELS/laser2/laser2.pt
VOCAB_FILE[0]=$MODELS/laser2/laser2.cvocab

BACKBONE_MODEL=roberta-base
MODEL_NAME[1]=roberta-student
MODEL_PATH[1]=$EXPERIMENTS/robust-embeddings/laser/experiment_024e/models
VOCAB_FILE[1]=$MODELS/$BACKBONE_MODEL/$BACKBONE_MODEL.cvocab

MODEL_NAME[2]=roberta-student-init
MODEL_PATH[2]=$EXPERIMENTS/robust-embeddings/laser/experiment_024f/models
VOCAB_FILE[2]=$MODELS/$BACKBONE_MODEL/$BACKBONE_MODEL.cvocab

MODEL_NAME[3]=character-roberta-student
MODEL_PATH[3]=$EXPERIMENTS/robust-embeddings/laser/experiment_024c/models
VOCAB_FILE[3]=$DATASETS/oscar/mini/4M/bin/charobertaugc-laserstd/dict.charobertaugc.txt

MODEL_NAME[4]=character-roberta-student-init
MODEL_PATH[4]=$EXPERIMENTS/robust-embeddings/laser/experiment_024d/models
VOCAB_FILE[4]=$DATASETS/oscar/mini/4M/bin/charobertaugc-laserstd/dict.charobertaugc.txt

VALIDATION_SEED=110
INPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_025
OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_025_valid
PREFIX=flores200/$VALIDATION_SEED/dev
#PREFIX=lexnorm2015/train

python $HOME/robust-embeddings/src/laser/validate.py \
    --input-tok-dir $INPUT_EXPERIMENT_DIR/tok \
    --teacher-model ${MODEL_NAME[0]} \
    --teacher-model-path ${MODEL_PATH[0]} \
    --teacher-vocab ${VOCAB_FILE[0]} \
    --student-model ${MODEL_NAME[$SLURM_ARRAY_TASK_ID]} \
    --student-model-dir ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]} \
    --student-vocab ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
    --ugc-file $PREFIX/cleaned.eng_Latn_ugc.dev \
    --std-file $PREFIX/cleaned.eng_Latn.dev \
    --output-dir $OUTPUT_EXPERIMENT_DIR \

echo "Done..."
