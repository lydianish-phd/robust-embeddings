#!/bin/bash

#SBATCH --job-name=eval-artificial      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/eval/eval_%j.log # Standard output and error log
#SBATCH --array=0-2

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile

MODEL_NAME[0]=laser
MODEL_PATH[0]=$MODELS/laser2/laser2.pt
VOCAB_FILE[0]=$MODELS/laser2/laser2.cvocab

MODEL_NAME[1]=roberta-student
MODEL_PATH[1]=$EXPERIMENTS/robust-embeddings/laser/experiment_024e/models/checkpoint_best.pt
VOCAB_FILE[1]=$MODELS/roberta-base/roberta-base.cvocab

MODEL_NAME[2]=roberta-student-init
MODEL_PATH[2]=$EXPERIMENTS/robust-embeddings/laser/experiment_024f/models/checkpoint_best.pt
VOCAB_FILE[2]=$MODELS/roberta-base/roberta-base.cvocab

MODEL_NAME[3]=character-roberta-student
MODEL_PATH[3]=$EXPERIMENTS/robust-embeddings/laser/experiment_024c/models/checkpoint_best.pt
VOCAB_FILE[3]=$DATASETS/oscar/mini/4M/bin/charobertaugc-laserastd/dict.charobertaugc.txt

MODEL_NAME[4]=character-roberta-student-init
MODEL_PATH[4]=$EXPERIMENTS/robust-embeddings/laser/experiment_024d/models/checkpoint_best.pt
VOCAB_FILE[4]=$DATASETS/oscar/mini/4M/bin/charobertaugc-laserstd/dict.charobertaugc.txt


#------------------ FLORES ------------------#

NOISY_LANGS="cleaned.eng_abr1_Latn,cleaned.eng_abr2_Latn,\
cleaned.eng_abr3_Latn,cleaned.eng_case_Latn,cleaned.eng_cont_Latn,\
cleaned.eng_dysl_Latn,cleaned.eng_fing_Latn,cleaned.eng_homo_Latn,\
cleaned.eng_leet_Latn,cleaned.eng_slng_Latn,cleaned.eng_spac_Latn,\
cleaned.eng_spel_Latn,cleaned.eng_week_Latn,cleaned.eng_Latn_ugc"
TARGET_LANG=cleaned.eng_Latn
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_025
INPUT_DIR=$EXPERIMENT_DIR/tok/${MODEL_NAME[$SLURM_ARRAY_TASK_ID]}
CORPUS=flores200
CORPUS_PARTS="dev"

for SEED in {100..109}
do
    for CORPUS_PART in $CORPUS_PARTS
    do
        OUTPUT_DIR=$INPUT_DIR/$CORPUS/$SEED/$CORPUS_PART
        
        echo "$CORPUS_PART"
        
        echo " - calculating xsim and cos dist"

        python3 $LASER/source/eval.py                \
            --base-dir $INPUT_DIR                         \
            --corpus $CORPUS/$SEED                        \
            --corpus-part $CORPUS_PART               \
            --margin ratio                           \
            --src-encoder ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]}  \
            --src-vocab-file ${VOCAB_FILE[$SLURM_ARRAY_TASK_ID]} \
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
            --src-langs $NOISY_LANGS      \
            --tgt-langs $TARGET_LANG \
            --tgt-aug-langs $TARGET_LANG \
            --output-dir $OUTPUT_DIR \
            --verbose 
    done
done
echo "Done..."