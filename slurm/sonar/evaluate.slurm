#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=18

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
MODELS="nllb1b sonar nllb600m rosonar"

# Multilingual

DATA_DIR[0]=$DATASETS/rocsmt/test
SRC_FILE[0]=${DATA_DIR[0]}/raw.en.test
REF_FILE[0]=${DATA_DIR[0]}/ref.fr.test
SYS_FILE_SUFFIX[0]=rocsmt/eng_Latn-fra_Latn/raw.en.test.out

DATA_DIR[1]=$DATASETS/rocsmt/test
SRC_FILE[1]=${DATA_DIR[1]}/norm.en.test
REF_FILE[1]=${DATA_DIR[1]}/ref.fr.test
SYS_FILE_SUFFIX[1]=rocsmt/eng_Latn-fra_Latn/norm.en.test.out

DATA_DIR[2]=$DATASETS/flores200/devtest
SRC_FILE[2]=${DATA_DIR[2]}/eng_Latn.devtest
REF_FILE[2]=${DATA_DIR[2]}/fra_Latn.devtest
SYS_FILE_SUFFIX[2]=flores200/eng_Latn-fra_Latn/eng_Latn.devtest.out

DATA_DIR[3]=$DATASETS/rocsmt/test
SRC_FILE[3]=${DATA_DIR[3]}/raw.en.test
REF_FILE[3]=${DATA_DIR[3]}/ref.de.test
SYS_FILE_SUFFIX[3]=rocsmt/eng_Latn-deu_Latn/raw.en.test.out

DATA_DIR[4]=$DATASETS/rocsmt/test
SRC_FILE[4]=${DATA_DIR[4]}/norm.en.test
REF_FILE[4]=${DATA_DIR[4]}/ref.de.test
SYS_FILE_SUFFIX[4]=rocsmt/eng_Latn-deu_Latn/norm.en.test.out

DATA_DIR[5]=$DATASETS/flores200/devtest
SRC_FILE[5]=${DATA_DIR[5]}/eng_Latn.devtest
REF_FILE[5]=${DATA_DIR[5]}/deu_Latn.devtest
SYS_FILE_SUFFIX[5]=flores200/eng_Latn-deu_Latn/eng_Latn.devtest.out

DATA_DIR[6]=$DATASETS/rocsmt/test
SRC_FILE[6]=${DATA_DIR[6]}/raw.en.test
REF_FILE[6]=${DATA_DIR[6]}/ref.cs.test
SYS_FILE_SUFFIX[6]=rocsmt/eng_Latn-ces_Latn/raw.en.test.out

DATA_DIR[7]=$DATASETS/rocsmt/test
SRC_FILE[7]=${DATA_DIR[7]}/norm.en.test
REF_FILE[7]=${DATA_DIR[7]}/ref.cs.test
SYS_FILE_SUFFIX[7]=rocsmt/eng_Latn-ces_Latn/norm.en.test.out

DATA_DIR[8]=$DATASETS/flores200/devtest
SRC_FILE[8]=${DATA_DIR[8]}/eng_Latn.devtest
REF_FILE[8]=${DATA_DIR[8]}/ces_Latn.devtest
SYS_FILE_SUFFIX[8]=flores200/eng_Latn-ces_Latn/eng_Latn.devtest.out

DATA_DIR[9]=$DATASETS/rocsmt/test
SRC_FILE[9]=${DATA_DIR[9]}/raw.en.test
REF_FILE[9]=${DATA_DIR[9]}/ref.uk.test
SYS_FILE_SUFFIX[9]=rocsmt/eng_Latn-ukr_Cyrl/raw.en.test.out

DATA_DIR[10]=$DATASETS/rocsmt/test
SRC_FILE[10]=${DATA_DIR[10]}/norm.en.test
REF_FILE[10]=${DATA_DIR[10]}/ref.uk.test
SYS_FILE_SUFFIX[10]=rocsmt/eng_Latn-ukr_Cyrl/norm.en.test.out

DATA_DIR[11]=$DATASETS/flores200/devtest
SRC_FILE[11]=${DATA_DIR[11]}/eng_Latn.devtest
REF_FILE[11]=${DATA_DIR[11]}/ukr_Cyrl.devtest
SYS_FILE_SUFFIX[11]=flores200/eng_Latn-ukr_Cyrl/eng_Latn.devtest.out

DATA_DIR[12]=$DATASETS/rocsmt/test
SRC_FILE[12]=${DATA_DIR[12]}/raw.en.test
REF_FILE[12]=${DATA_DIR[12]}/ref.ru.test
SYS_FILE_SUFFIX[12]=rocsmt/eng_Latn-rus_Cyrl/raw.en.test.out

DATA_DIR[13]=$DATASETS/rocsmt/test
SRC_FILE[13]=${DATA_DIR[13]}/norm.en.test
REF_FILE[13]=${DATA_DIR[13]}/ref.ru.test
SYS_FILE_SUFFIX[13]=rocsmt/eng_Latn-rus_Cyrl/norm.en.test.out

DATA_DIR[14]=$DATASETS/flores200/devtest
SRC_FILE[14]=${DATA_DIR[14]}/eng_Latn.devtest
REF_FILE[14]=${DATA_DIR[14]}/rus_Cyrl.devtest
SYS_FILE_SUFFIX[14]=flores200/eng_Latn-rus_Cyrl/eng_Latn.devtest.out

# Standard bilingual

DATA_DIR[15]=$DATASETS/flores200/devtest
SRC_FILE[15]=${DATA_DIR[15]}/fra_Latn.devtest
REF_FILE[15]=${DATA_DIR[15]}/eng_Latn.devtest
SYS_FILE_SUFFIX[15]=flores200/fra_Latn-eng_Latn/fra_Latn.devtest.out

DATA_DIR[16]=$DATASETS/wmt2015
SRC_FILE[16]=${DATA_DIR[16]}/newsdiscusstest2015-enfr-src.en
REF_FILE[16]=${DATA_DIR[16]}/newsdiscusstest2015-enfr-ref.fr
SYS_FILE_SUFFIX[16]=wmt2015/eng_Latn-fra_Latn/newsdiscusstest2015-enfr-src.en.out

DATA_DIR[17]=$DATASETS/wmt2015
SRC_FILE[17]=${DATA_DIR[17]}/newsdiscusstest2015-enfr-ref.fr
REF_FILE[17]=${DATA_DIR[17]}/newsdiscusstest2015-enfr-src.en
SYS_FILE_SUFFIX[17]=wmt2015/fra_Latn-eng_Latn/newsdiscusstest2015-enfr-ref.fr.out

# Non-standard English

DATA_DIR[18]=$DATASETS/footweets
SRC_FILE[18]=${DATA_DIR[18]}/detok.twitter.sent.en.txt
REF_FILE[18]=${DATA_DIR[18]}/detok.twitter.sent.de.txt
SYS_FILE_SUFFIX[18]=footweets/eng_Latn-deu_Latn/detok.twitter.sent.en.txt.out

DATA_DIR[19]=$DATASETS/mtnt/MTNT2019
SRC_FILE[19]=${DATA_DIR[19]}/en-fr.en
REF_FILE[19]=${DATA_DIR[19]}/en-fr.fr
SYS_FILE_SUFFIX[19]=mtnt/eng_Latn-fra_Latn/en-fr.en.out

DATA_DIR[20]=$DATASETS/mtnt/MTNT2019
SRC_FILE[20]=${DATA_DIR[20]}/en-ja.en
REF_FILE[20]=${DATA_DIR[20]}/en-ja.ja
SYS_FILE_SUFFIX[20]=mtnt/eng_Latn-jpn_Jpan/en-ja.en.out

# Non-standard French

DATA_DIR[21]=$DATASETS/mmtc
SRC_FILE[21]=${DATA_DIR[21]}/test.fr-en.fr
REF_FILE[21]=${DATA_DIR[21]}/test.fr-en.en
SYS_FILE_SUFFIX[21]=mmtc/fra_Latn-eng_Latn/test.fr-en.fr.out

DATA_DIR[22]=$DATASETS/mtnt/MTNT2019
SRC_FILE[22]=${DATA_DIR[22]}/fr-en.fr
REF_FILE[22]=${DATA_DIR[22]}/fr-en.en
SYS_FILE_SUFFIX[22]=mtnt/fra_Latn-eng_Latn/fr-en.fr.out

DATA_DIR[23]=$DATASETS/pfsmb/tmp
SRC_FILE[23]=${DATA_DIR[23]}/test.fr
REF_FILE[23]=${DATA_DIR[23]}/test.en
SYS_FILE_SUFFIX[23]=pfsmb/fra_Latn-eng_Latn/test.fr.out

# Autoencode

DATA_DIR[24]=$DATASETS/rocsmt/test
SRC_FILE[24]=${DATA_DIR[24]}/ref.fr.test
REF_FILE[24]=${DATA_DIR[24]}/ref.fr.test
SYS_FILE_SUFFIX[24]=rocsmt/fra_Latn/ref.fr.test.out

DATA_DIR[25]=$DATASETS/rocsmt/test
SRC_FILE[25]=${DATA_DIR[25]}/norm.en.test
REF_FILE[25]=${DATA_DIR[25]}/norm.en.test
SYS_FILE_SUFFIX[25]=rocsmt/eng_Latn/norm.en.test.out

DATA_DIR[26]=$DATASETS/rocsmt/test
SRC_FILE[26]=${DATA_DIR[26]}/raw.en.test
REF_FILE[26]=${DATA_DIR[26]}/raw.en.test
SYS_FILE_SUFFIX[26]=rocsmt/eng_Latn/raw.en.test.out

DATA_DIR[27]=$DATASETS/rocsmt/test
SRC_FILE[27]=${DATA_DIR[27]}/raw.en.test
REF_FILE[27]=${DATA_DIR[27]}/norm.en.test
SYS_FILE_SUFFIX[27]=rocsmt/eng_Latn-eng_Latn/raw.en.test.out

SYS_FILES=()

for MODEL in $MODELS
do
    echo "Model: $MODEL"
        
    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL
    SYS_FILES+=("$OUTPUT_DIR/${SYS_FILE_SUFFIX[$SLURM_ARRAY_TASK_ID]}")
done

python $HOME/robust-embeddings/src/sonar/evaluate.py \
    --src-file ${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
    --sys-files "${SYS_FILES[@]}" \
    --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]}

echo "Done..."

