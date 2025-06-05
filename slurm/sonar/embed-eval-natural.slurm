#!/bin/bash

#SBATCH --job-name=embed-eval-natural      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=20	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:30:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1-2

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD #anaconda-py3/2022.10

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047t

MODEL[0]=sonar
MODEL[1]=rosonar
MODEL[2]=rosonar_std
MODEL[3]=rolaser_v2

#------------------ LEXNORM ------------------#

CORPUS[0]=multilexnorm2021/en
CORPUS_PARTS[0]="test"
SRC_LANGS[0]="en.src"
TGT_LANGS[0]="en.ref"
TGT_AUG_LANGS[0]=""

#------------------ ROCSMT ------------------#

CORPUS[1]=rocsmt
CORPUS_PARTS[1]="test"
SRC_LANGS[1]="raw.en"
TGT_LANGS[1]="norm.en"
TGT_AUG_LANGS[1]=""

EMBED_DIR_PREFIX=$EXPERIMENT_DIR/embeddings/${MODEL[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/scores/${MODEL[$SLURM_ARRAY_TASK_ID]}

echo "Model: ${MODEL[$SLURM_ARRAY_TASK_ID]}"

for i in {0..1} # loop through corpora
do
    for CORPUS_PART in ${CORPUS_PARTS[$i]}
    do
        EMBED_DIR=$EMBED_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART
        OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${CORPUS[$i]}/$CORPUS_PART

        echo "${CORPUS[$i]} $CORPUS_PART"
        echo " - calculating xsim(++)"

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
            --offline-embeddings \
            --verbose 

    done
done
echo "Done..."
