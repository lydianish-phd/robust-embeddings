#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=32	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=2-4

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

#module load anaconda-py3/2023.09 pytorch-gpu/py3/2.3.0
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
MODEL=sonar

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/rocsmt/test
INPUT_FILES[0]="${DATA_DIR[0]}/raw.en.test ${DATA_DIR[0]}/norm.en.test"
SRC_LANG[0]=eng_Latn
TGT_LANG[0]=fra_Latn

CORPUS[1]=rocsmt
DATA_DIR[1]=$DATASETS/rocsmt/test
INPUT_FILES[1]="${DATA_DIR[1]}/raw.en.test ${DATA_DIR[1]}/norm.en.test"
SRC_LANG[1]=eng_Latn
TGT_LANG[1]=deu_Latn

CORPUS[2]=rocsmt
DATA_DIR[2]=$DATASETS/rocsmt/test
INPUT_FILES[2]="${DATA_DIR[2]}/raw.en.test ${DATA_DIR[2]}/norm.en.test"
SRC_LANG[2]=eng_Latn
TGT_LANG[2]=ces_Latn

CORPUS[3]=rocsmt
DATA_DIR[3]=$DATASETS/rocsmt/test
INPUT_FILES[3]="${DATA_DIR[3]}/raw.en.test ${DATA_DIR[3]}/norm.en.test"
SRC_LANG[3]=eng_Latn
TGT_LANG[3]=ukr_Cyrl

CORPUS[4]=rocsmt
DATA_DIR[4]=$DATASETS/rocsmt/test
INPUT_FILES[4]="${DATA_DIR[4]}/raw.en.test ${DATA_DIR[4]}/norm.en.test"
SRC_LANG[4]=eng_Latn
TGT_LANG[4]=rus_Cyrl

LANG_PAIR=${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL/${CORPUS[$SLURM_ARRAY_TASK_ID]}/$LANG_PAIR

for INPUT_FILE in ${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
do
    python $HOME/robust-embeddings/src/sonar/generate-sonar.py \
        -i $INPUT_FILE \
        -s ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
        -t ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
        -o $OUTPUT_DIR
done

echo "Done..."

