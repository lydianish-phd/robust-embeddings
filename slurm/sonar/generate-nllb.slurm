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
#SBATCH --time=3:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=5

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd anaconda-py3/2022.10 #pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc 
source $HOME/.bash_profile
source $HOME/.bash_python_exports

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047

MODEL=nllb600m
MODEL_PATH=$MODELS/nllb600m/nllb200densedst600mcheckpoint

#MODEL=nllb1b
#MODEL_PATH=$MODELS/nllb1b/nllb200dense1bcheckpoint

SPM_MODEL_PATH=$MODELS/nllb200-spm/flores200sacrebleuspm
SPM_MODEL_DICT=$MODELS/nllb200-spm/nllb200dictionary

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
INPUT_FILES[13]="${DATA_DIR[13]}/twitter.sent.en.txt"
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
DATA_DIR[18]=$DATASETS/pfsmb/tmp
INPUT_FILES[18]="${DATA_DIR[18]}/test.fr"
SRC_LANG[18]=fra_Latn
TGT_LANG[18]=eng_Latn

# Autoencode

CORPUS[19]=rocsmt
DATA_DIR[19]=$DATASETS/rocsmt/test
INPUT_FILES[19]="${DATA_DIR[19]}/ref.fr.test"
SRC_LANG[19]=fra_Latn
TGT_LANG[19]=fra_Latn

CORPUS[20]=rocsmt
DATA_DIR[20]=$DATASETS/rocsmt/test
INPUT_FILES[20]="${DATA_DIR[20]}/norm.en.test"
SRC_LANG[20]=eng_Latn
TGT_LANG[20]=eng_Latn

CORPUS[21]=rocsmt
DATA_DIR[21]=$DATASETS/rocsmt/test
INPUT_FILES[21]="${DATA_DIR[21]}/raw.en.test"
SRC_LANG[21]=eng_Latn
TGT_LANG[21]=eng_Latn

# Output dir

OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/outputs/$MODEL/${CORPUS[$SLURM_ARRAY_TASK_ID]}

if [ "$SLURM_ARRAY_TASK_ID" -ge 19 ] # autoencode
then 
    OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}
    BIN_DIR=${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/bin/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}
else
    OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}
    BIN_DIR=${DATA_DIR[$SLURM_ARRAY_TASK_ID]}/bin/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}
fi

# Languages ($FLORES200_LANGS and $LANG_PAIR)

source $HOME/robust-embeddings/src/sonar/flores200.langs.txt 
LANG_PAIR=${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}

# Launch the generation of outputs

for INPUT_FILE in ${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
do
    INPUT_FILE_TOK=$INPUT_FILE.tok.${SRC_LANG[$SLURM_ARRAY_TASK_ID]}

    if [ ! -e "$INPUT_FILE_TOK" ]
    then
        echo "Tokenizing input data..."
        python $FAIRSEQ_PATH/scripts/spm_encode.py \
            --model $SPM_MODEL_PATH \
            --output_format piece < $INPUT_FILE > $INPUT_FILE_TOK
    fi
    
    if [ ! -e "$BIN_DIR/test.$LANG_PAIR.${SRC_LANG[$SLURM_ARRAY_TASK_ID]}.bin" ]
    then
        echo "Binarizing input data..."
        fairseq-preprocess \
            --source-lang ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
            --target-lang ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
            --testpref $INPUT_FILE.tok \
            --destdir $BIN_DIR \
            --srcdict $SPM_MODEL_DICT \
            --tgtdict $SPM_MODEL_DICT \
            --only-source \
            --workers 16
        
        cp $BIN_DIR/dict.${SRC_LANG[$SLURM_ARRAY_TASK_ID]}.txt $BIN_DIR/dict.${TGT_LANG[$SLURM_ARRAY_TASK_ID]}.txt
    fi
    
    echo "Generating outputs..."
    INPUT_FILE_NAME=$(basename "$INPUT_FILE")

    fairseq-generate $BIN_DIR \
        --path $MODEL_PATH \
        --task translation_multi_simple_epoch \
        --langs $FLORES200_LANGS \
        --lang-pairs $LANG_PAIR \
        --source-lang ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
        --target-lang ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
        --encoder-langtok tgt \
        --decoder-langtok \
        --keep-inference-langtok \
        --gen-subset test \
        --batch-size 32 \
        --beam 5 \
        --nbest 1 \
        --fp16 \
        --bpe sentencepiece \
        --sentencepiece-model $SPM_MODEL_PATH \
        --remove-bpe \
        --results-path $OUTPUT_DIR

    if [ "$SLURM_ARRAY_TASK_ID" -eq 21 ] # for evaluating against normalised reference
    then
        NEW_OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL/${CORPUS[$SLURM_ARRAY_TASK_ID]}/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${SRC_LANG[$SLURM_ARRAY_TASK_ID]}
        mkdir -p $NEW_OUTPUT_DIR
        cp $OUTPUT_DIR/raw.en.test.out $NEW_OUTPUT_DIR
    fi
done

echo "Done..."

