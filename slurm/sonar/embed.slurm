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
##SBATCH --array=0-2

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 AND accelerate
module load arch/a100 pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047s

MODEL[0]=sonar
MODEL_DIR[0]=""

MODEL[1]=rosonar
MODEL_DIR[1]="-m $EXPERIMENT_DIR/models/rosonar/checkpoint-70000"

MODEL[2]=rosonar_std
MODEL_DIR[2]="-m $EXPERIMENT_DIR/models/rosonar_std/checkpoint-70000"


#------------------ LEXNORM ------------------#

CORPUS[0]=multilexnorm2021/en
CORPUS_PARTS[0]="test"
SRC_LANGS[0]="en.src"
TGT_LANGS[0]="en.ref"
TGT_AUG_LANGS[0]=""

#------------------ ROCSMT ------------------#

CORPUS[1]=rocsmt
CORPUS_PARTS[1]="test"
SRC_LANGS[1]="raw.en,ref.fr,ref.de,ref.cs,ref.uk,ref.ru"
TGT_LANGS[1]="norm.en"
TGT_AUG_LANGS[1]=""

#------------------ FLORES ------------------#

CORPUS[2]=flores200
CORPUS_PARTS[2]="devtest"
SRC_LANGS[2]="fra_Latn,deu_Latn,ces_Latn,ukr_Cyrl,rus_Cyrl"
TGT_LANGS[2]="eng_Latn"
TGT_AUG_LANGS[2]=""

EMBED_DIR=$EXPERIMENT_DIR/embeddings/${MODEL[$SLURM_ARRAY_TASK_ID]}
mkdir -p $EMBED_DIR

for i in {0..2}
do
    echo "Model: ${MODEL[$SLURM_ARRAY_TASK_ID]}"
    
    for CORPUS_PART in ${CORPUS_PARTS[$i]}
    do

        echo " - embedding $CORPUS $CORPUS_PART"
        
        python $HOME/robust-embeddings/src/sonar/embed.py \
            --data-dir $DATASETS \
            --embed-dir $EMBED_DIR \
            --corpus ${CORPUS[$i]} \
            --split $CORPUS_PART \
            --src-langs ${SRC_LANGS[$i]} \
            --tgt-langs ${TGT_LANGS[$i]} \
            ${MODEL_DIR[$SLURM_ARRAY_TASK_ID]} \
    
    done
done

echo "Done..."

