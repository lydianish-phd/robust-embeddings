#!/bin/bash

#SBATCH --job-name=eval-artificial      # Job name
#SBATCH --account=ncm@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
MODEL_NAME[0]=laser
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py

EXPERIMENTS_DIR=$EXPERIMENTS/robust-embeddings/laser
DATASETS_DIR=$DATASETS/oscar/mini/4Mlrec/bin

MODEL_NAME[1]=roberta-maxpool
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_026/models/checkpoint_best.pt
VOCAB_FILE[1]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=roberta-maxpool-init
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_026b/models/checkpoint_best.pt
VOCAB_FILE[2]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=roberta-meanpool
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_026c/models/checkpoint_best.pt
VOCAB_FILE[3]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[4]=roberta-meanpool-init
MODEL_PATH[4]=$EXPERIMENTS_DIR/experiment_026d/models/checkpoint_best.pt
VOCAB_FILE[4]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[4]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[5]=c-roberta-maxpool
MODEL_PATH[5]=$EXPERIMENTS_DIR/experiment_026e/models/checkpoint_best.pt
VOCAB_FILE[5]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[5]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[6]=c-roberta-maxpool-init
MODEL_PATH[6]=$EXPERIMENTS_DIR/experiment_026f/models/checkpoint_best.pt
VOCAB_FILE[6]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[6]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[7]=c-roberta-meanpool
MODEL_PATH[7]=$EXPERIMENTS_DIR/experiment_026g/models/checkpoint_best.pt
VOCAB_FILE[7]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[7]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[8]=c-roberta-meanpool-init
MODEL_PATH[8]=$EXPERIMENTS_DIR/experiment_026h/models/checkpoint_best.pt
VOCAB_FILE[8]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[8]=$HOME/data-preparation/src/char-tokenizer.py

#------------------ FLORES ------------------#

NOISY_LANGS="eng_abr1_Latn,eng_abr2_Latn,\
eng_abr3_Latn,eng_cont_Latn,\
eng_dysl_Latn,eng_fing_Latn,eng_homo_Latn,\
eng_leet_Latn,eng_slng_Latn,eng_spac_Latn,\
eng_spel_Latn,eng_week_Latn,eng_Latn_ugc"
TARGET_LANG=eng_Latn
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_027
OUTPUT_DIR=$EXPERIMENT_DIR/scores/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}
CORPUS=flores200
CORPUS_PARTS="dev"

for SEED in {100..109}
do
    echo "Seed $SEED"
    for CORPUS_PART in $CORPUS_PARTS
    do
        INPUT_DIR=$DATASETS/$CORPUS/ugc/$SEED/$CORPUS_PART
        OUTPUT_DIR=$OUTPUT_DIR/$CORPUS/$SEED/$CORPUS_PART
        
        echo "$CORPUS_PART"
        
        echo " - calculating xsim and cos dist"

        python3 $LASER/source/eval.py                \
            --base-dir $INPUT_DIR                         \
            --corpus $CORPUS/$SEED                        \
            --corpus-part $CORPUS_PART               \
            --margin ratio                           \
            --src-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
            --src-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
            --src-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
            --src-langs $NOISY_LANGS      \
            --tgt-langs $TARGET_LANG \
            --output-dir $OUTPUT_DIR \
            --cosine-distances \
            --verbose 

        echo " - calculating xsim++"

        python3 $LASER/source/eval.py                \
            --base-dir $INPUT_DIR                         \
            --corpus $CORPUS/$SEED                         \
            --corpus-part $CORPUS_PART               \
            --margin ratio                           \
            --src-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
            --src-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
            --src-tokenizer ${TOKENIZER[$SLURM_ARRAY_TASK_ID]} \
            --src-langs $NOISY_LANGS      \
            --tgt-langs $TARGET_LANG \
            --tgt-aug-langs $TARGET_LANG \
            --output-dir $OUTPUT_DIR \
            --verbose 
    done
done
echo "Done..."