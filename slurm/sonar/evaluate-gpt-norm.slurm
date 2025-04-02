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
#SBATCH --array=0,3,6,9,12,18-24

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

SPM_MODEL=$MODELS/nllb600m/flores200sacrebleuspm

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r
MODELS="nllb600m"
# MODELS="rosonar" # rosonar_std"

# Multilingual

DATA_DIR[0]=$DATASETS/rocsmt/test
SRC_FILE[0]=${DATA_DIR[0]}/gpt.raw.en.test
REF_FILE[0]=${DATA_DIR[0]}/ref.fr.test
SYS_FILE_SUFFIX[0]=rocsmt/eng_Latn-fra_Latn/gpt.raw.en.test.out

DATA_DIR[3]=$DATASETS/rocsmt/test
SRC_FILE[3]=${DATA_DIR[3]}/gpt.raw.en.test
REF_FILE[3]=${DATA_DIR[3]}/ref.de.test
SYS_FILE_SUFFIX[3]=rocsmt/eng_Latn-deu_Latn/gpt.raw.en.test.out

DATA_DIR[6]=$DATASETS/rocsmt/test
SRC_FILE[6]=${DATA_DIR[6]}/gpt.raw.en.test
REF_FILE[6]=${DATA_DIR[6]}/ref.cs.test
SYS_FILE_SUFFIX[6]=rocsmt/eng_Latn-ces_Latn/gpt.raw.en.test.out

DATA_DIR[9]=$DATASETS/rocsmt/test
SRC_FILE[9]=${DATA_DIR[9]}/gpt.raw.en.test
REF_FILE[9]=${DATA_DIR[9]}/ref.uk.test
SYS_FILE_SUFFIX[9]=rocsmt/eng_Latn-ukr_Cyrl/gpt.raw.en.test.out

DATA_DIR[12]=$DATASETS/rocsmt/test
SRC_FILE[12]=${DATA_DIR[12]}/gpt.raw.en.test
REF_FILE[12]=${DATA_DIR[12]}/ref.ru.test
SYS_FILE_SUFFIX[12]=rocsmt/eng_Latn-rus_Cyrl/gpt.raw.en.test.out

# Non-standard English

DATA_DIR[18]=$DATASETS/footweets
SRC_FILE[18]=${DATA_DIR[18]}/gpt.detok.twitter.sent.en.txt
REF_FILE[18]=${DATA_DIR[18]}/detok.twitter.sent.de.txt
SYS_FILE_SUFFIX[18]=footweets/eng_Latn-deu_Latn/gpt.detok.twitter.sent.en.txt.out

DATA_DIR[19]=$DATASETS/mtnt/MTNT2019
SRC_FILE[19]=${DATA_DIR[19]}/gpt.en-fr.en
REF_FILE[19]=${DATA_DIR[19]}/en-fr.fr
SYS_FILE_SUFFIX[19]=mtnt/eng_Latn-fra_Latn/gpt.en-fr.en.out

DATA_DIR[20]=$DATASETS/mtnt/MTNT2019
SRC_FILE[20]=${DATA_DIR[20]}/gpt.en-ja.en
REF_FILE[20]=${DATA_DIR[20]}/en-ja.ja
SYS_FILE_SUFFIX[20]=mtnt/eng_Latn-jpn_Jpan/gpt.en-ja.en.out

# Non-standard French

DATA_DIR[21]=$DATASETS/mmtc
SRC_FILE[21]=${DATA_DIR[21]}/gpt.test.fr-en.fr
REF_FILE[21]=${DATA_DIR[21]}/test.fr-en.en
SYS_FILE_SUFFIX[21]=mmtc/fra_Latn-eng_Latn/gpt.test.fr-en.fr.out

DATA_DIR[22]=$DATASETS/mtnt/MTNT2019
SRC_FILE[22]=${DATA_DIR[22]}/gpt.fr-en.fr
REF_FILE[22]=${DATA_DIR[22]}/fr-en.en
SYS_FILE_SUFFIX[22]=mtnt/fra_Latn-eng_Latn/gpt.fr-en.fr.out

DATA_DIR[23]=$DATASETS/pfsmb
SRC_FILE[23]=${DATA_DIR[23]}/gpt.test.fr
REF_FILE[23]=${DATA_DIR[23]}/test.en
SYS_FILE_SUFFIX[23]=pfsmb/fra_Latn-eng_Latn/gpt.test.fr.out

DATA_DIR[24]=$DATASETS/foursquare
SRC_FILE[24]=${DATA_DIR[24]}/gpt.test.fr
REF_FILE[24]=${DATA_DIR[24]}/test.en
SYS_FILE_SUFFIX[24]=foursquare/fra_Latn-eng_Latn/gpt.test.fr.out

SYS_FILES=()


echo "Models: $MODELS"

for MODEL in $MODELS
do
    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL
    SYS_FILES+=("$OUTPUT_DIR/${SYS_FILE_SUFFIX[$SLURM_ARRAY_TASK_ID]}")
done

python $HOME/robust-embeddings/src/sonar/evaluate.py \
    --src-file ${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
    --sys-files "${SYS_FILES[@]}" \
    --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]}

# python $HOME/robust-embeddings/src/sonar/stats.py \
#     -i "${SYS_FILES[@]}" \
#     -m $SPM_MODEL

echo "Done..."

