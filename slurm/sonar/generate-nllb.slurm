#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2
##SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:20:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-22

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using anaconda-py3/2023.09 which has pytorch 2.2.2 but NO accelerate (otherwise fails to load NLLB pretrained model from HF)
# module load arch/a100 anaconda-py3/2023.09 libsndfile/1.0.28 intel-oneapi-tbb/2021.9
module load pytorch-gpu/py3/2.0.0 #V100

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r

MODEL[0]=nllb600m
MODEL_NAME[0]=facebook/nllb-200-distilled-600m
MODEL_PATH[0]=facebook/nllb-200-distilled-600m

MODEL[1]=nllb1b
MODEL_NAME[1]=facebook/nllb-200-distilled-1.3B
MODEL_PATH[1]=facebook/nllb-200-distilled-1.3B

MODEL[2]=nllb3b
MODEL_NAME[2]=facebook/nllb-200-3.3B
MODEL_PATH[2]=facebook/nllb-200-3.3B

MODEL[3]=nllb600m_ft_nat_en_fr
MODEL_NAME[3]=facebook/nllb-200-distilled-600m
MODEL_PATH[3]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/natural/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[4]=nllb600m_ft_nat_fr_en
MODEL_NAME[4]=facebook/nllb-200-distilled-600m
MODEL_PATH[4]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/natural/fra_Latn-eng_Latn/nllb600m/checkpoint_best

MODEL[5]=nllb600m_ft_syn_en_fr
MODEL_NAME[5]=facebook/nllb-200-distilled-600m
MODEL_PATH[5]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/synthetic/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[6]=nllb600m_ft_syn_plus_en_fr
MODEL_NAME[6]=facebook/nllb-200-distilled-600m
MODEL_PATH[6]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/synthetic_plus/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[7]=nllb600m_ft_syn_plus_en_fr_last
MODEL_NAME[7]=facebook/nllb-200-distilled-600m
MODEL_PATH[7]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/synthetic_plus/eng_Latn-fra_Latn/nllb600m/checkpoint_last

# Multilingual

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

CORPUS[5]=flores200
DATA_DIR[5]=$DATASETS/flores200/devtest
INPUT_FILES[5]="${DATA_DIR[5]}/eng_Latn.devtest"
SRC_LANG[5]=eng_Latn
TGT_LANG[5]=fra_Latn

CORPUS[6]=flores200
DATA_DIR[6]=$DATASETS/flores200/devtest
INPUT_FILES[6]="${DATA_DIR[6]}/eng_Latn.devtest"
SRC_LANG[6]=eng_Latn
TGT_LANG[6]=deu_Latn

CORPUS[7]=flores200
DATA_DIR[7]=$DATASETS/flores200/devtest
INPUT_FILES[7]="${DATA_DIR[7]}/eng_Latn.devtest"
SRC_LANG[7]=eng_Latn
TGT_LANG[7]=ces_Latn

CORPUS[8]=flores200
DATA_DIR[8]=$DATASETS/flores200/devtest
INPUT_FILES[8]="${DATA_DIR[8]}/eng_Latn.devtest"
SRC_LANG[8]=eng_Latn
TGT_LANG[8]=ukr_Cyrl

CORPUS[9]=flores200
DATA_DIR[9]=$DATASETS/flores200/devtest
INPUT_FILES[9]="${DATA_DIR[9]}/eng_Latn.devtest"
SRC_LANG[9]=eng_Latn
TGT_LANG[9]=rus_Cyrl

# Standard bilingual

CORPUS[10]=flores200
DATA_DIR[10]=$DATASETS/flores200/devtest
INPUT_FILES[10]="${DATA_DIR[10]}/fra_Latn.devtest"
SRC_LANG[10]=fra_Latn
TGT_LANG[10]=eng_Latn

CORPUS[11]=wmt2015
DATA_DIR[11]=$DATASETS/wmt2015
INPUT_FILES[11]="${DATA_DIR[11]}/newsdiscusstest2015-enfr-src.en"
SRC_LANG[11]=eng_Latn
TGT_LANG[11]=fra_Latn

CORPUS[12]=wmt2015
DATA_DIR[12]=$DATASETS/wmt2015
INPUT_FILES[12]="${DATA_DIR[12]}/newsdiscusstest2015-enfr-ref.fr"
SRC_LANG[12]=fra_Latn
TGT_LANG[12]=eng_Latn

# Non-standard English

CORPUS[13]=footweets
DATA_DIR[13]=$DATASETS/footweets
INPUT_FILES[13]="${DATA_DIR[13]}/detok.twitter.sent.en.txt"
SRC_LANG[13]=eng_Latn
TGT_LANG[13]=deu_Latn

CORPUS[14]=mtnt
DATA_DIR[14]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[14]="${DATA_DIR[14]}/en-fr.en"
SRC_LANG[14]=eng_Latn
TGT_LANG[14]=fra_Latn

CORPUS[15]=mtnt
DATA_DIR[15]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[15]="${DATA_DIR[15]}/en-ja.en"
SRC_LANG[15]=eng_Latn
TGT_LANG[15]=jpn_Jpan

# Non-standard French

CORPUS[16]=mmtc
DATA_DIR[16]=$DATASETS/mmtc
INPUT_FILES[16]="${DATA_DIR[16]}/test.fr-en.fr"
SRC_LANG[16]=fra_Latn
TGT_LANG[16]=eng_Latn

CORPUS[17]=mtnt
DATA_DIR[17]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[17]="${DATA_DIR[17]}/fr-en.fr"
SRC_LANG[17]=fra_Latn
TGT_LANG[17]=eng_Latn

CORPUS[18]=pfsmb
DATA_DIR[18]=$DATASETS/pfsmb
INPUT_FILES[18]="${DATA_DIR[18]}/test.fr"
SRC_LANG[18]=fra_Latn
TGT_LANG[18]=eng_Latn

CORPUS[19]=foursquare
DATA_DIR[19]=$DATASETS/foursquare
INPUT_FILES[19]="${DATA_DIR[19]}/test.fr"
SRC_LANG[19]=fra_Latn
TGT_LANG[19]=eng_Latn

# Autoencode

CORPUS[20]=rocsmt
DATA_DIR[20]=$DATASETS/rocsmt/test
INPUT_FILES[20]="${DATA_DIR[20]}/ref.fr.test"
SRC_LANG[20]=fra_Latn
TGT_LANG[20]=fra_Latn

CORPUS[21]=rocsmt
DATA_DIR[21]=$DATASETS/rocsmt/test
INPUT_FILES[21]="${DATA_DIR[21]}/norm.en.test"
SRC_LANG[21]=eng_Latn
TGT_LANG[21]=eng_Latn

CORPUS[22]=rocsmt
DATA_DIR[22]=$DATASETS/rocsmt/test
INPUT_FILES[22]="${DATA_DIR[22]}/raw.en.test"
SRC_LANG[22]=eng_Latn
TGT_LANG[22]=eng_Latn

for i in {7..7}
do
    echo "Model: ${MODEL[$i]}"

    OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/outputs/${MODEL[$i]}/${CORPUS[$SLURM_ARRAY_TASK_ID]}

    if [ "$SLURM_ARRAY_TASK_ID" -ge 20 ] # autoencode
    then 
        OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]} 
    else
        OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}
    fi

    # Launch the generation of outputs
    for INPUT_FILE in ${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
    do 
        echo -e "\t - input file: $INPUT_FILE"

        python $HOME/robust-embeddings/src/sonar/generate-nllb.py \
            -i $INPUT_FILE \
            -s ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
            -t ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
            -o $OUTPUT_DIR \
            -m ${MODEL_NAME[$i]} \
            -p ${MODEL_PATH[$i]}

        if [ "$SLURM_ARRAY_TASK_ID" -eq 22 ] # for evaluating against normalised reference
        then
            NEW_OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${SRC_LANG[$SLURM_ARRAY_TASK_ID]}
            mkdir -p $NEW_OUTPUT_DIR
            cp $OUTPUT_DIR/raw.en.test.out $NEW_OUTPUT_DIR
        fi
    done
done

echo "Done..."

