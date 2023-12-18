#!/bin/bash

#SBATCH --job-name=eval-multilingual      # Job name
#SBATCH --account=ncm@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-1

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_BASE_NUM="032"

MODEL_NAME[0]=laser
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py

EXPERIMENTS_DIR=$EXPERIMENTS/robust-embeddings/laser
DATASETS_DIR=$DATASETS/oscar/mini/4Mlrec/bin

MODEL_NAME[1]=roberta-maxpool
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}/models/checkpoint_best.pt
VOCAB_FILE[1]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=roberta-maxpool-init
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}b/models/checkpoint_best.pt
VOCAB_FILE[2]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=roberta-meanpool
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}c/models/checkpoint_best.pt
VOCAB_FILE[3]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[4]=roberta-meanpool-init
MODEL_PATH[4]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}d/models/checkpoint_best.pt
VOCAB_FILE[4]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[4]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[5]=c-roberta-maxpool
MODEL_PATH[5]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}e/models/checkpoint_best.pt
VOCAB_FILE[5]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[5]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[6]=c-roberta-maxpool-init
MODEL_PATH[6]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}f/models/checkpoint_best.pt
VOCAB_FILE[6]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[6]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[7]=c-roberta-meanpool
MODEL_PATH[7]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}g/models/checkpoint_best.pt
VOCAB_FILE[7]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[7]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[8]=c-roberta-meanpool-init
MODEL_PATH[8]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}h/models/checkpoint_best.pt
VOCAB_FILE[8]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[8]=$HOME/data-preparation/src/char-tokenizer.py

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_eval
OUTPUT_DIR_PREFIX=$OUTPUT_EXPERIMENT_DIR/scores/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}

CORPUS=rocsmt
CORPUS_PART="test"
PIVOT_LANGS="raw.en,norm.en"
OTHER_LANGS="ref.fr,ref.de,ref.cs,ref.uk,ref.ru"
 
#CORPUS=flores200
#CORPUS_PART="dev"
#PIVOT_LANGS="eng_Latn"
#OTHER_LANGS="fra_Latn,deu_Latn,ces_Latn,ukr_Cyrl,rus_Cyrl"

echo "xx-en"

echo " - calculating xsim and cos dist"

OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$CORPUS/$CORPUS_PART/xx-en

python3 $LASER/source/eval.py                \
    --base-dir $DATASETS                         \
    --corpus $CORPUS                        \
    --corpus-part $CORPUS_PART               \
    --margin ratio                           \
    --src-encoder ${MODEL_PATH[0]} \
    --src-vocab-file ${VOCAB_FILE[0]} \
    --src-tokenizer ${TOKENIZER[0]} \
    --src-langs $OTHER_LANGS \
    --tgt-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
    --tgt-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
    --tgt-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
    --tgt-langs $PIVOT_LANGS       \
    --output-dir $OUTPUT_DIR \
    --cosine-distances \
    --verbose 

if [ "$CORPUS" = "flores200" ]
then
    echo " - calculating xsim++"

    OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$CORPUS/$CORPUS_PART/xx-en

    python3 $LASER/source/eval.py                \
        --base-dir $DATASETS                         \
        --corpus $CORPUS                        \
        --corpus-part $CORPUS_PART               \
        --margin ratio                           \
        --src-encoder ${MODEL_PATH[0]} \
        --src-vocab-file ${VOCAB_FILE[0]} \
        --src-tokenizer ${TOKENIZER[0]} \
        --src-langs $OTHER_LANGS \
        --tgt-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
        --tgt-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
        --tgt-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
        --tgt-langs $PIVOT_LANGS       \
        --tgt-aug-langs $PIVOT_LANGS \
        --output-dir $OUTPUT_DIR \
        --verbose
fi

echo "en-xx"

echo " - calculating xsim and cos dist"

OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$CORPUS/$CORPUS_PART/en-xx

python3 $LASER/source/eval.py                \
    --base-dir $DATASETS                         \
    --corpus $CORPUS                        \
    --corpus-part $CORPUS_PART               \
    --margin ratio                           \
    --src-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
    --src-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
    --src-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
    --src-langs $PIVOT_LANGS      \
    --tgt-encoder ${MODEL_PATH[0]} \
    --tgt-vocab-file ${VOCAB_FILE[0]} \
    --tgt-tokenizer ${TOKENIZER[0]} \
    --tgt-langs $OTHER_LANGS \
    --output-dir $OUTPUT_DIR \
    --cosine-distances \
    --verbose 
