#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=28

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
OUTPUT_DIR=$EXPERIMENT_DIR/outputs
SCORES_DIR=$EXPERIMENT_DIR/scores/comet_compare
mkdir -p $SCORES_DIR

TOPLINE_MODEL="nllb1b"
MODELS="sonar nllb600m"

# Autoencode

CORPUS[28]=rocsmt
DATA_DIR[28]=$DATASETS/${CORPUS[28]}/test
SRC_FILE_NAME[28]=raw.en.test
REF_FILE_NAME[28]=norm.en.test
LANG_PAIR[28]=eng_Latn-eng_Latn


# Comparing scores

SRC_FILE=${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/${SRC_FILE_NAME[$SLURM_ARRAY_TASK_ID]}
REF_FILE=${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/${REF_FILE_NAME[$SLURM_ARRAY_TASK_ID]}
SYS_FILE_SUFFIX=${CORPUS[$SLURM_ARRAY_TASK_ID]}/${LANG_PAIR[$SLURM_ARRAY_TASK_ID]}/${SRC_FILE_NAME[$SLURM_ARRAY_TASK_ID]}.out

for MODEL in $MODELS
do 
    echo "Comparing scores between $TOPLINE_MODEL $MODEL"

    SYS_FILES=("$OUTPUT_DIR/$TOPLINE_MODEL/$SYS_FILE_SUFFIX" "$OUTPUT_DIR/$MODEL/$SYS_FILE_SUFFIX")

    comet-compare --model Unbabel/wmt22-comet-da \
        -s ${SRC_FILE} \
        -r ${REF_FILE} \
        -t ${SYS_FILES[@]} \
        --num_workers 20 \
        --only_system \
        --to_json $SCORES_DIR/${TOPLINE_MODEL}-${MODEL}_${CORPUS[$SLURM_ARRAY_TASK_ID]}_${LANG_PAIR[$SLURM_ARRAY_TASK_ID]}_${SRC_FILE_NAME[$SLURM_ARRAY_TASK_ID]}.out.json
done

echo "Done..."

