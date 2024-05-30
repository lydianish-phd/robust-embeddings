#!/bin/bash

#SBATCH --job-name=eval-natural      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=9

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

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

MODEL_NAME[1]=roberta-maxpool-init-0.1
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}/models/checkpoint_best.pt
VOCAB_FILE[1]=$DATASETS_DIR/4M_10/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=roberta-maxpool-init-0.2
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}b/models/checkpoint_best.pt
VOCAB_FILE[2]=$DATASETS_DIR/4M_20/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=roberta-maxpool-init-0.3
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}c/models/checkpoint_best.pt
VOCAB_FILE[3]=$DATASETS_DIR/4M_30/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[4]=roberta-maxpool-init-0.4
MODEL_PATH[4]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}d/models/checkpoint_best.pt
VOCAB_FILE[4]=$DATASETS_DIR/4M_40/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[4]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[5]=roberta-maxpool-init-0.5
MODEL_PATH[5]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}e/models/checkpoint_best.pt
VOCAB_FILE[5]=$DATASETS_DIR/4M_50/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[5]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[6]=roberta-maxpool-init-0
MODEL_PATH[6]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}f/models/checkpoint_best.pt
VOCAB_FILE[6]=$DATASETS_DIR/4M_0/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[6]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[7]=roberta-maxpool-init-0.35
MODEL_PATH[7]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}g/models/checkpoint_best.pt
VOCAB_FILE[7]=$DATASETS_DIR/4M_35/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[7]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[8]=roberta-maxpool-init-0.45
MODEL_PATH[8]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}h/models/checkpoint_best.pt
VOCAB_FILE[8]=$DATASETS_DIR/4M_45/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[8]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[9]=roberta-maxpool-init-0.6
MODEL_PATH[9]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}i/models/checkpoint_best.pt
VOCAB_FILE[9]=$DATASETS_DIR/4M_60/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[9]=$HOME/data-preparation/src/roberta-tokenizer.py

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_eval
OUTPUT_DIR_PREFIX=$OUTPUT_EXPERIMENT_DIR/scores/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}

#------------------ LEXNORM ------------------#

NOISY_LANGS[0]=en.src
TARGET_LANG[0]=en.ref
CORPUS[0]=multilexnorm2021/en
CORPUS_PARTS[0]="test"

#------------------ ROCSMT ------------------#

NOISY_LANGS[1]=raw.en
TARGET_LANG[1]=norm.en
CORPUS[1]=rocsmt
CORPUS_PARTS[1]="test"

#------------------ ACCIDENTS ------------------#

NOISY_LANGS[2]=src
TARGET_LANG[2]=ref
CORPUS[2]=accidents
CORPUS_PARTS[2]="train test"

for i in {0..2}
do
    echo "${CORPUS[$i]}"
    for CORPUS_PART in ${CORPUS_PARTS[$i]}
    do
        OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART

        echo "$CORPUS_PART"
        
        echo " - calculating xsim and cos dist"

        python3 $LASER/source/eval.py                \
            --base-dir $DATASETS                         \
            --corpus ${CORPUS[$i]}                        \
            --corpus-part $CORPUS_PART               \
            --margin ratio                           \
            --src-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
            --src-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
            --src-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
            --src-langs ${NOISY_LANGS[$i]}      \
            --tgt-langs ${TARGET_LANG[$i]} \
            --output-dir $OUTPUT_DIR \
            --cosine-distances \
            --verbose 

    done
done
echo "Done..."
