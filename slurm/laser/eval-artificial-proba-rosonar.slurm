#!/bin/bash

#SBATCH --job-name=eval-natural      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=6:00:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1-3

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_BASE_NUM="048"

MODEL_NAME[0]=LASER
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py

EXPERIMENTS_DIR=$EXPERIMENTS/robust-embeddings/laser
DATASETS_DIR=$DATASETS/rosonar/monolingual/rolaser/eng

MODEL_NAME[1]=RoLASER-0.1
MODEL_PATH[1]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}/models/checkpoint_last.pt
VOCAB_FILE[1]=$DATASETS_DIR/10/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=RoLASER-0.2
MODEL_PATH[2]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}b/models/checkpoint_last.pt
VOCAB_FILE[2]=$DATASETS_DIR/20/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=RoLASER-0.3
MODEL_PATH[3]=$EXPERIMENTS_DIR/experiment_${EXPERIMENT_BASE_NUM}c/models/checkpoint_last.pt
VOCAB_FILE[3]=$DATASETS_DIR/30/bin/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/roberta-tokenizer.py

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_eval
OUTPUT_DIR_PREFIX=$OUTPUT_EXPERIMENT_DIR/scores/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}

#------------------ FLORES ------------------#

# NOISY_LANGS="eng_abr1_Latn,eng_abr2_Latn,\
# eng_abr3_Latn,eng_cont_Latn,\
# eng_dysl_Latn,eng_fing_Latn,eng_homo_Latn,\
# eng_leet_Latn,eng_slng_Latn,eng_spac_Latn,\
# eng_spel_Latn,eng_week_Latn,eng_Latn_mix_all"
NOISY_LANGS="eng_Latn_mix_all"
TARGET_LANG=eng_Latn
OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_BASE_NUM}_eval
OUTPUT_DIR_PREFIX=$OUTPUT_EXPERIMENT_DIR/scores/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}
CORPUS=flores200
CORPUS_PART=dev
PROBAS="0.1 0.2 0.3 0.4 0.5"

for SEED in {1005..1009}
do
    echo "Seed $SEED"
    OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$CORPUS/$SEED/$CORPUS_PART
    
    for PROBA in $PROBAS
    do
        echo "$PROBA"

        cp -r $DATASETS/$CORPUS/${CORPUS_PART}/$TARGET_LANG.$CORPUS_PART $DATASETS/$CORPUS/artificial/$SEED/$PROBA/ugc/$CORPUS_PART
        cp -r $DATASETS/$CORPUS/${CORPUS_PART}_augmented $DATASETS/$CORPUS/artificial/$SEED/$PROBA/ugc
        
        echo " - calculating xsim++"

        python3 $LASER/source/eval.py                \
            --base-dir $DATASETS                         \
            --corpus $CORPUS/artificial/$SEED/$PROBA/ugc                         \
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
