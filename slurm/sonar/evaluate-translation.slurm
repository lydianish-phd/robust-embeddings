#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=1:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

#module load anaconda-py3/2023.09 pytorch-gpu/py3/2.3.0
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
MODEL=sonar
OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL


DATA_DIR[0]=$DATASETS/rocsmt/test
SRC_FILE[0]=${DATA_DIR[0]}/raw.en.test
REF_FILE[0]=${DATA_DIR[0]}/ref.fr.test
SYS_FILE[0]=$OUTPUT_DIR/rocsmt/eng_Latn-fra_Latn/raw.en.test.out

DATA_DIR[1]=$DATASETS/rocsmt/test
SRC_FILE[1]=${DATA_DIR[1]}/norm.en.test
REF_FILE[1]=${DATA_DIR[1]}/ref.fr.test
SYS_FILE[1]=$OUTPUT_DIR/rocsmt/eng_Latn-fra_Latn/norm.en.test.out

DATA_DIR[2]=$DATASETS/flores200/devtest
SRC_FILE[2]=${DATA_DIR[2]}/eng_Latn.devtest
REF_FILE[2]=${DATA_DIR[2]}/fra_Latn.devtest
SYS_FILE[2]=$OUTPUT_DIR/flores200/eng_Latn-fra_Latn/eng_Latn.devtest.out


for INPUT_FILE in ${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
do
    python $HOME/robust-embeddings/src/sonar/evaluate-translation.py \
        --src-file ${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
        --sys-file ${SYS_FILE[$SLURM_ARRAY_TASK_ID]} \
        --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]} 
done

echo "Done..."

