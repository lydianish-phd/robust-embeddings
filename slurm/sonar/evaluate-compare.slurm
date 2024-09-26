#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=1:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-28

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
INPUT_DIR=$EXPERIMENT_DIR/outputs
SCORES_DIR=$EXPERIMENT_DIR/scores/comet_compare

TOPLINE_MODEL="nllb1b"
MODELS="sonar nllb600m"

# Multilingual

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/${CORPUS[0]}/test
SRC_FILE_NAME[0]=raw.en.test
REF_FILE_NAME[0]=ref.fr.test
LANG_PAIR[0]=eng_Latn-fra_Latn

CORPUS[1]=rocsmt
DATA_DIR[1]=$DATASETS/${CORPUS[1]}/test
SRC_FILE_NAME[1]=norm.en.test
REF_FILE_NAME[1]=ref.fr.test
LANG_PAIR[1]=eng_Latn-fra_Latn

CORPUS[2]=flores200
DATA_DIR[2]=$DATASETS/${CORPUS[2]}/devtest
SRC_FILE_NAME[2]=eng_Latn.devtest
REF_FILE_NAME[2]=fra_Latn.devtest
LANG_PAIR[2]=eng_Latn-fra_Latn

CORPUS[3]=rocsmt
DATA_DIR[3]=$DATASETS/${CORPUS[3]}/test
SRC_FILE_NAME[3]=raw.en.test
REF_FILE_NAME[3]=ref.de.test
LANG_PAIR[3]=eng_Latn-deu_Latn

CORPUS[4]=rocsmt
DATA_DIR[4]=$DATASETS/${CORPUS[4]}/test
SRC_FILE_NAME[4]=norm.en.test
REF_FILE_NAME[4]=ref.de.test
LANG_PAIR[4]=eng_Latn-deu_Latn

CORPUS[5]=flores200
DATA_DIR[5]=$DATASETS/${CORPUS[5]}/devtest
SRC_FILE_NAME[5]=eng_Latn.devtest
REF_FILE_NAME[5]=deu_Latn.devtest
LANG_PAIR[5]=eng_Latn-deu_Latn

CORPUS[6]=rocsmt
DATA_DIR[6]=$DATASETS/${CORPUS[6]}/test
SRC_FILE_NAME[6]=raw.en.test
REF_FILE_NAME[6]=ref.cs.test
LANG_PAIR[6]=eng_Latn-ces_Latn

CORPUS[7]=rocsmt
DATA_DIR[7]=$DATASETS/${CORPUS[7]}/test
SRC_FILE_NAME[7]=norm.en.test
REF_FILE_NAME[7]=ref.cs.test
LANG_PAIR[7]=eng_Latn-ces_Latn

CORPUS[8]=flores200
DATA_DIR[8]=$DATASETS/${CORPUS[8]}/devtest
SRC_FILE_NAME[8]=eng_Latn.devtest
REF_FILE_NAME[8]=ces_Latn.devtest
LANG_PAIR[8]=eng_Latn-ces_Latn

CORPUS[9]=rocsmt
DATA_DIR[9]=$DATASETS/${CORPUS[9]}/test
SRC_FILE_NAME[9]=raw.en.test
REF_FILE_NAME[9]=ref.uk.test
LANG_PAIR[9]=eng_Latn-ukr_Cyrl

CORPUS[10]=rocsmt
DATA_DIR[10]=$DATASETS/${CORPUS[10]}/test
SRC_FILE_NAME[10]=norm.en.test
REF_FILE_NAME[10]=ref.uk.test
LANG_PAIR[10]=eng_Latn-ukr_Cyrl

CORPUS[11]=flores200
DATA_DIR[11]=$DATASETS/${CORPUS[11]}/devtest
SRC_FILE_NAME[11]=eng_Latn.devtest
REF_FILE_NAME[11]=ukr_Cyrl.devtest
LANG_PAIR[11]=eng_Latn-ukr_Cyrl

CORPUS[12]=rocsmt
DATA_DIR[12]=$DATASETS/${CORPUS[12]}/test
SRC_FILE_NAME[12]=raw.en.test
REF_FILE_NAME[12]=ref.ru.test
LANG_PAIR[12]=eng_Latn-rus_Cyrl

CORPUS[13]=rocsmt
DATA_DIR[13]=$DATASETS/${CORPUS[13]}/test
SRC_FILE_NAME[13]=norm.en.test
REF_FILE_NAME[13]=ref.ru.test
LANG_PAIR[13]=eng_Latn-rus_Cyrl

CORPUS[14]=flores200
DATA_DIR[14]=$DATASETS/${CORPUS[14]}/devtest
SRC_FILE_NAME[14]=eng_Latn.devtest
REF_FILE_NAME[14]=rus_Cyrl.devtest
LANG_PAIR[14]=eng_Latn-rus_Cyrl

# Standard bilingual

CORPUS[15]=flores200
DATA_DIR[15]=$DATASETS/${CORPUS[15]}/devtest
SRC_FILE_NAME[15]=fra_Latn.devtest
REF_FILE_NAME[15]=eng_Latn.devtest
LANG_PAIR[15]=fra_Latn-eng_Latn

CORPUS[16]=wmt2015
DATA_DIR[16]=$DATASETS/${CORPUS[16]}
SRC_FILE_NAME[16]=newsdiscusstest2015-enfr-src.en
REF_FILE_NAME[16]=newsdiscusstest2015-enfr-ref.fr
LANG_PAIR[16]=eng_Latn-fra_Latn

CORPUS[17]=wmt2015
DATA_DIR[17]=$DATASETS/${CORPUS[17]}
SRC_FILE_NAME[17]=newsdiscusstest2015-enfr-ref.fr
REF_FILE_NAME[17]=newsdiscusstest2015-enfr-src.en
LANG_PAIR[17]=fra_Latn-eng_Latn

# Non-standard English

CORPUS[18]=footweets
DATA_DIR[18]=$DATASETS/${CORPUS[18]}
SRC_FILE_NAME[18]=detok.twitter.sent.en.txt
REF_FILE_NAME[18]=detok.twitter.sent.de.txt
LANG_PAIR[18]=eng_Latn-deu_Latn

CORPUS[19]=mtnt
DATA_DIR[19]=$DATASETS/${CORPUS[19]}/MTNT2019
SRC_FILE_NAME[19]=en-fr.en
REF_FILE_NAME[19]=en-fr.fr
LANG_PAIR[19]=eng_Latn-fra_Latn

CORPUS[20]=mtnt
DATA_DIR[20]=$DATASETS/${CORPUS[20]}/MTNT2019
SRC_FILE_NAME[20]=en-ja.en
REF_FILE_NAME[20]=en-ja.ja
LANG_PAIR[20]=eng_Latn-jpn_Jpan

# Non-standard French

CORPUS[21]=mmtc
DATA_DIR[21]=$DATASETS/${CORPUS[21]}
SRC_FILE_NAME[21]=test.fr-en.fr
REF_FILE_NAME[21]=test.fr-en.en
LANG_PAIR[21]=fra_Latn-eng_Latn

CORPUS[22]=mtnt
DATA_DIR[22]=$DATASETS/${CORPUS[22]}/MTNT2019
SRC_FILE_NAME[22]=fr-en.fr
REF_FILE_NAME[22]=fr-en.en
LANG_PAIR[22]=fra_Latn-eng_Latn

CORPUS[23]=pfsmb
DATA_DIR[23]=$DATASETS/${CORPUS[23]}
SRC_FILE_NAME[23]=test.fr
REF_FILE_NAME[23]=test.en
LANG_PAIR[23]=fra_Latn-eng_Latn

CORPUS[24]=foursquare
DATA_DIR[24]=$DATASETS/${CORPUS[24]}
SRC_FILE_NAME[24]=test.fr
REF_FILE_NAME[24]=test.en
LANG_PAIR[24]=fra_Latn-eng_Latn

# Autoencode

for i in {25..28}
do
    CORPUS[$i]=rocsmt
    DATA_DIR[$i]=$DATASETS/${CORPUS[$i]}/test
done

SRC_FILE_NAME[25]=ref.fr.test
REF_FILE_NAME[25]=ref.fr.test
LANG_PAIR[25]=fra_Latn

SRC_FILE_NAME[26]=norm.en.test
REF_FILE_NAME[26]=norm.en.test
LANG_PAIR[26]=eng_Latn

SRC_FILE_NAME[27]=raw.en.test
REF_FILE_NAME[27]=raw.en.test
LANG_PAIR[27]=eng_Latn

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

    SYS_FILES=("$INPUT_DIR/$TOPLINE_MODEL/$SYS_FILE_SUFFIX" "$INPUT_DIR/$MODEL/$SYS_FILE_SUFFIX")
    OUTPUT_DIR=$SCORES_DIR/${TOPLINE_MODEL}-${MODEL}
    mkdir -p $OUTPUT_DIR

    comet-compare --model Unbabel/wmt22-comet-da \
        -s ${SRC_FILE} \
        -r ${REF_FILE} \
        -t ${SYS_FILES[@]} \
        --num_workers 20 \
        --only_system \
        --to_json $OUTPUT_DIR/${CORPUS[$SLURM_ARRAY_TASK_ID]}_${LANG_PAIR[$SLURM_ARRAY_TASK_ID]}_${SRC_FILE_NAME[$SLURM_ARRAY_TASK_ID]}.out.json
done

echo "Done..."

