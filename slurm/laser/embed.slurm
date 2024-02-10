#!/bin/bash

#SBATCH --job-name=embed      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=6:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0,1,5

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

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}_eval
OUTPUT_DIR_PREFIX=$OUTPUT_EXPERIMENT_DIR/embeddings/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}

#------------------ ROCSMT ------------------#

LANGS[0]="raw.en norm.en ref.cs ref.de ref.fr ref.ru ref.uk"
for i in {1..8}
do
    LANGS[$i]="raw.en norm.en"
done

CORPUS=rocsmt
CORPUS_PART=test

OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$CORPUS/$CORPUS_PART
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/* # overwrite

for LANG in ${LANGS[$SLURM_ARRAY_TASK_ID]}
do
    INPUT_FILE_NAME=$DATASETS/$CORPUS/$CORPUS_PART/$LANG.$CORPUS_PART
    OUTPUT_EMBED_FILE=$OUTPUT_DIR/$LANG.$CORPUS_PART.bin
    
    echo "Embedding $INPUT_FILE_NAME with ${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}..."

    python $LASER/source/embed.py \
        --input     $INPUT_FILE_NAME        \
        --encoder   ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}    \
        --custom-vocab-file   ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]}    \
        --custom-tokenizer   ${TOKENIZER[$SLURM_ARRAY_TASK_ID]}    \
        --output    $OUTPUT_EMBED_FILE       \
        --verbose

done
echo "Done..."

