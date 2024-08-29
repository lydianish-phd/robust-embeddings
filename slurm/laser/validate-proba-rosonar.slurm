#!/bin/bash

#SBATCH --job-name=validate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=6:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=9

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_BASE_NUM="048"
EXPERIMENTS_DIR=$EXPERIMENTS/robust-embeddings/laser
DATASETS_DIR=$DATASETS/rosonar/monolingual/rolaser/eng
VALID_FILE_BASENAME=oscar2301_eng_decontam_part2_sub_valid

MODEL_NAME[0]=LASER
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py
VALID_FILE[0]=$DATASETS_DIR/10/${VALID_FILE_BASENAME}_chunks/orig/${VALID_FILE_BASENAME}-0.txt

MODEL_NAME[1]=RoLASER-0.1
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}/models
VOCAB_FILE[1]=$DATASETS_DIR/10/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py
VALID_FILE[1]=$DATASETS_DIR/10/${VALID_FILE_BASENAME}_chunks/ugc/${VALID_FILE_BASENAME}-0.txt

MODEL_NAME[2]=RoLASER-0.2
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}b/models
VOCAB_FILE[2]=$DATASETS_DIR/20/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py
VALID_FILE[2]=$DATASETS_DIR/20/${VALID_FILE_BASENAME}_chunks/ugc/${VALID_FILE_BASENAME}-0.txt

MODEL_NAME[3]=RoLASER-0.3
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}c/models
VOCAB_FILE[3]=$DATASETS_DIR/30/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py
VALID_FILE[3]=$DATASETS_DIR/30/${VALID_FILE_BASENAME}_chunks/ugc/${VALID_FILE_BASENAME}-0.txt

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_valid

python $HOME/robust-embeddings/src/laser/validate.py \
    --teacher-model ${MODEL_NAME[0]} \
    --teacher-model-path ${MODEL_PATH[0]} \
    --teacher-vocab ${VOCAB_FILE[0]} \
    --teacher-tok ${TOKENIZER[0]} \
    --student-model ${MODEL_NAME[$SLURM_ARRAY_TASK_ID]} \
    --student-model-dir ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]} \
    --student-vocab ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
    --student-tok ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
    --ugc-file ${VALID_FILE[$SLURM_ARRAY_TASK_ID]} \
    --std-file ${VALID_FILE[0]} \
    --output-dir $OUTPUT_EXPERIMENT_DIR \

echo "Done..."
