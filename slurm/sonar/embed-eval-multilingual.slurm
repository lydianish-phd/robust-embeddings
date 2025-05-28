#!/bin/bash

#SBATCH --job-name=embed-eval-multilingual      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=20	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-3

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047s

MODEL[0]=sonar
EMBEDDINGS[0]="--offline-embeddings"

MODEL[1]=rosonar
EMBEDDINGS[1]="--offline-embeddings"

MODEL[2]=rosonar_std
EMBEDDINGS[2]="--offline-embeddings"

MODEL[3]=lydianish/RoLASER-v2
EMBEDDINGS[3]="--use-hugging-face"

#------------------ FLORES ------------------#

CORPUS[0]=flores200
CORPUS_PARTS[0]="devtest"
SRC_LANGS[0]="fra_Latn" #,deu_Latn,ces_Latn,ukr_Cyrl,rus_Cyrl"
TGT_LANGS[0]="eng_Latn"
TGT_AUG_LANGS[0]="--tgt-aug-langs eng_Latn"

#------------------ ROCSMT ------------------#

CORPUS[1]=rocsmt
CORPUS_PARTS[1]="test"
SRC_LANGS[1]="ref.fr" #,ref.de,ref.cs,ref.uk,ref.ru"
TGT_LANGS[1]="raw.en,norm.en"
TGT_AUG_LANGS[1]=""


EMBED_DIR_PREFIX=$EXPERIMENT_DIR/embeddings/${MODEL[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/scores/${MODEL[$SLURM_ARRAY_TASK_ID]}

echo "Model: ${MODEL[$SLURM_ARRAY_TASK_ID]}"

for i in {0..1} # loop through corpora
do
    for CORPUS_PART in ${CORPUS_PARTS[$i]}
    do
        echo "${CORPUS[$i]} $CORPUS_PART xx-en"
        echo " - calculating xsim(++)" # and cos dist"

        EMBED_DIR=$EMBED_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART
        OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART/xx-en

        python3 $LASER/source/eval.py \
            --base-dir $DATASETS \
            --embed-dir $EMBED_DIR \
            --corpus ${CORPUS[$i]} \
            --corpus-part $CORPUS_PART \
            --margin ratio \
            --src-encoder ${MODEL[$SLURM_ARRAY_TASK_ID]} \
            --src-langs ${SRC_LANGS[$i]} \
            --tgt-langs ${TGT_LANGS[$i]} \
            ${TGT_AUG_LANGS[$i]} \
            --output-dir $OUTPUT_DIR \
            ${EMBEDDINGS[$SLURM_ARRAY_TASK_ID]} \
            --verbose \
            # --cosine-distances \

        # echo "${CORPUS[$i]} $CORPUS_PART en-xx"
        # echo " - calculating xsim and cos dist"

        # OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART/en-xx

        # python3 $LASER/source/eval.py \
        #     --base-dir $DATASETS \
        #     --embed-dir $EMBED_DIR \
        #     --corpus ${CORPUS[$i]} \
        #     --corpus-part $CORPUS_PART \
        #     --margin ratio \
        #     --src-encoder ${MODEL[$SLURM_ARRAY_TASK_ID]} \
        #     --src-langs ${TGT_LANGS[$i]} \
        #     --tgt-langs ${SRC_LANGS[$i]} \
        #     --output-dir $OUTPUT_DIR \
        #     --cosine-distances \
        #     --offline-embeddings \
        #     --verbose 
    done
done
echo "Done..."
