#!/bin/bash

#SBATCH --job-name=embed      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=32	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=2:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-3

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 AND accelerate
module load arch/a100 pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047s

MODEL[0]=sonar
MODEL_PATH[0]="text_sonar_basic_encoder"

MODEL[1]=rosonar
MODEL_PATH[1]="$EXPERIMENT_DIR/models/rosonar/checkpoint-70000"

MODEL[2]=rosonar_std
MODEL_PATH[2]="$EXPERIMENT_DIR/models/rosonar_std/checkpoint-70000"

MODEL[3]=rolaser_v2
MODEL_PATH[3]="lydianish/RoLASER-v2"

#------------------ LEXNORM ------------------#

CORPUS[0]=multilexnorm2021/en
CORPUS_PARTS[0]="test"
SRC_LANGS[0]="en.src"
TGT_LANGS[0]="en.ref"
TGT_AUG_LANGS[0]=""
SEEDS[0]="None"
NOISE_PROBAS[0]="None"

#------------------ ROCSMT ------------------#

CORPUS[1]=rocsmt
CORPUS_PARTS[1]="test"
SRC_LANGS[1]="raw.en,ref.fr,ref.de,ref.cs,ref.uk,ref.ru"
TGT_LANGS[1]="norm.en"
TGT_AUG_LANGS[1]=""
SEEDS[1]="None"
NOISE_PROBAS[1]="None"

#------------------ FLORES ------------------#

CORPUS[2]=flores200
CORPUS_PARTS[2]="devtest"
SRC_LANGS[2]="fra_Latn,deu_Latn,ces_Latn,ukr_Cyrl,rus_Cyrl"
TGT_LANGS[2]="eng_Latn"
TGT_AUG_LANGS[2]="eng_Latn"
SEEDS[2]="None"
NOISE_PROBAS[2]="None"

#------------------ FLORES AUGMENTED ------------------#

CORPUS[3]=flores200/ugc
CORPUS_PARTS[3]="devtest"
SRC_LANGS[3]="eng_abr1_Latn,eng_abr2_Latn,\
eng_abr3_Latn,eng_cont_Latn,\
eng_dysl_Latn,eng_fing_Latn,eng_homo_Latn,\
eng_leet_Latn,eng_slng_Latn,eng_spac_Latn,\
eng_spel_Latn,eng_week_Latn,eng_Latn_mix_all"
TGT_LANGS[3]="eng_Latn"
TGT_AUG_LANGS[3]="eng_Latn"
SEEDS[3]="100 101 102 103 104 105 106 107 108 109"
NOISE_PROBAS[3]="None"

#------------------ ROCSMT AUGMENTED ------------------#

CORPUS[4]=rocsmt/artificial
CORPUS_PARTS[4]="test"
SRC_LANGS[4]="raw.en,norm.en_mix_all"
TGT_LANGS[4]="norm.en"
TGT_AUG_LANGS[4]=""
SEEDS[4]="1000 1001 1002 1003 1004"
NOISE_PROBAS[4]="0.05 0.1"

EMBED_DIR=$EXPERIMENT_DIR/embeddings/${MODEL[$SLURM_ARRAY_TASK_ID]}
mkdir -p $EMBED_DIR

echo "Model: ${MODEL[$SLURM_ARRAY_TASK_ID]}"

for i in {0..4} # loop through corpora
do    
    for CORPUS_PART in ${CORPUS_PARTS[$i]}
    do

        echo " - embedding ${CORPUS[$i]} $CORPUS_PART"
        for SEED in ${SEEDS[$i]}
        do 
            if [[ "$SEED" = "None" ]]
            then
                SEED=""
            else
                echo "Seed $SEED"
            fi
            for NOISE_PROBA in ${NOISE_PROBAS[$i]}
            do
                if [[ "${NOISE_PROBAS[$i]}" = "None" ]]
                then
                    PROBA=""
                else
                    echo "Proba $NOISE_PROBA"
                    PROBA=$NOISE_PROBA/ugc
                fi
                python $HOME/robust-embeddings/src/sonar/embed.py \
                    --model ${MODEL[$SLURM_ARRAY_TASK_ID]} \
                    --model-path ${MODEL_PATH[$SLURM_ARRAY_TASK_ID]} \
                    --data-dir $DATASETS \
                    --embed-dir $EMBED_DIR \
                    --corpus ${CORPUS[$i]}/$SEED/$PROBA \
                    --split $CORPUS_PART \
                    --src-langs ${SRC_LANGS[$i]} \
                    --tgt-langs ${TGT_LANGS[$i]} \
                    --tgt-aug-langs ${TGT_AUG_LANGS[$i]} \

            done            
        done
    done
done

echo "Done..."

