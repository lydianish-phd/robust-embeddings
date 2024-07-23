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

# Multilingual

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

DATA_DIR[3]=$DATASETS/rocsmt/test
SRC_FILE[3]=${DATA_DIR[3]}/raw.en.test
REF_FILE[3]=${DATA_DIR[3]}/ref.de.test
SYS_FILE[3]=$OUTPUT_DIR/rocsmt/eng_Latn-deu_Latn/raw.en.test.out

DATA_DIR[4]=$DATASETS/rocsmt/test
SRC_FILE[4]=${DATA_DIR[4]}/norm.en.test
REF_FILE[4]=${DATA_DIR[4]}/ref.de.test
SYS_FILE[4]=$OUTPUT_DIR/rocsmt/eng_Latn-deu_Latn/norm.en.test.out

DATA_DIR[5]=$DATASETS/flores200/devtest
SRC_FILE[5]=${DATA_DIR[5]}/eng_Latn.devtest
REF_FILE[5]=${DATA_DIR[5]}/deu_Latn.devtest
SYS_FILE[5]=$OUTPUT_DIR/flores200/eng_Latn-deu_Latn/eng_Latn.devtest.out

DATA_DIR[6]=$DATASETS/rocsmt/test
SRC_FILE[6]=${DATA_DIR[6]}/raw.en.test
REF_FILE[6]=${DATA_DIR[6]}/ref.cs.test
SYS_FILE[6]=$OUTPUT_DIR/rocsmt/eng_Latn-ces_Latn/raw.en.test.out

DATA_DIR[7]=$DATASETS/rocsmt/test
SRC_FILE[7]=${DATA_DIR[7]}/norm.en.test
REF_FILE[7]=${DATA_DIR[7]}/ref.cs.test
SYS_FILE[7]=$OUTPUT_DIR/rocsmt/eng_Latn-ces_Latn/norm.en.test.out

DATA_DIR[8]=$DATASETS/flores200/devtest
SRC_FILE[8]=${DATA_DIR[8]}/eng_Latn.devtest
REF_FILE[8]=${DATA_DIR[8]}/ces_Latn.devtest
SYS_FILE[8]=$OUTPUT_DIR/flores200/eng_Latn-ces_Latn/eng_Latn.devtest.out

DATA_DIR[9]=$DATASETS/rocsmt/test
SRC_FILE[9]=${DATA_DIR[9]}/raw.en.test
REF_FILE[9]=${DATA_DIR[9]}/ref.uk.test
SYS_FILE[9]=$OUTPUT_DIR/rocsmt/eng_Latn-ukr_Cyrl/raw.en.test.out

DATA_DIR[10]=$DATASETS/rocsmt/test
SRC_FILE[10]=${DATA_DIR[10]}/norm.en.test
REF_FILE[10]=${DATA_DIR[10]}/ref.uk.test
SYS_FILE[10]=$OUTPUT_DIR/rocsmt/eng_Latn-ukr_Cyrl/norm.en.test.out

DATA_DIR[11]=$DATASETS/flores200/devtest
SRC_FILE[11]=${DATA_DIR[11]}/eng_Latn.devtest
REF_FILE[11]=${DATA_DIR[11]}/ukr_Cyrl.devtest
SYS_FILE[11]=$OUTPUT_DIR/flores200/eng_Latn-ukr_Cyrl/eng_Latn.devtest.out

DATA_DIR[12]=$DATASETS/rocsmt/test
SRC_FILE[12]=${DATA_DIR[12]}/raw.en.test
REF_FILE[12]=${DATA_DIR[12]}/ref.ru.test
SYS_FILE[12]=$OUTPUT_DIR/rocsmt/eng_Latn-rus_Cyrl/raw.en.test.out

DATA_DIR[13]=$DATASETS/rocsmt/test
SRC_FILE[13]=${DATA_DIR[13]}/norm.en.test
REF_FILE[13]=${DATA_DIR[13]}/ref.ru.test
SYS_FILE[13]=$OUTPUT_DIR/rocsmt/eng_Latn-rus_Cyrl/norm.en.test.out

DATA_DIR[14]=$DATASETS/flores200/devtest
SRC_FILE[14]=${DATA_DIR[14]}/eng_Latn.devtest
REF_FILE[14]=${DATA_DIR[14]}/rus_Cyrl.devtest
SYS_FILE[14]=$OUTPUT_DIR/flores200/eng_Latn-rus_Cyrl/eng_Latn.devtest.out

python $HOME/robust-embeddings/src/sonar/evaluate-translation.py \
        --src-file ${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
        --sys-file ${SYS_FILE[$SLURM_ARRAY_TASK_ID]} \
        --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]} 

echo "Done..."

