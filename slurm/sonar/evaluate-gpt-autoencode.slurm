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
#SBATCH --array=27-28

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

SPM_MODEL=$MODELS/nllb600m/flores200sacrebleuspm

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r
MODEL="gpt4omini"

# Autoencode

DATA_DIR[27]=$DATASETS/rocsmt/test
SRC_FILE[27]=raw.en.test
REF_FILE[27]=raw.en.test
SYS_FILE_PREFIX[27]=rocsmt/eng_Latn/gpt.raw.en.test

DATA_DIR[28]=$DATASETS/rocsmt/test
SRC_FILE[28]=raw.en.test
REF_FILE[28]=norm.en.test
SYS_FILE_PREFIX[28]=rocsmt/eng_Latn-eng_Latn

SYS_FILES=()


echo "Model: $MODEL"

OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL/${SYS_FILE_PREFIX[$SLURM_ARRAY_TASK_ID]}
GPT_FILE="gpt.${SRC_FILE[$SLURM_ARRAY_TASK_ID]}"

echo "Copying $GPT_FILE to $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
cp ${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/$GPT_FILE $OUTPUT_DIR/$GPT_FILE


SYS_FILES+=("$OUTPUT_DIR/$GPT_FILE")

python $HOME/robust-embeddings/src/sonar/evaluate.py \
    --src-file ${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
    --sys-files "${SYS_FILES[@]}" \
    --ref-file ${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/${REF_FILE[$SLURM_ARRAY_TASK_ID]}

# python $HOME/robust-embeddings/src/sonar/stats.py \
#     -i "${SYS_FILES[@]}" \
#     -m $SPM_MODEL

echo "Done..."

