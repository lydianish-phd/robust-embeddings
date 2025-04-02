#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4,13-19

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

# MODEL[1]=nllb1b
# MODEL_NAME[1]=facebook/nllb-200-distilled-1.3B

# MODEL[2]=nllb3b
# MODEL_NAME[2]=facebook/nllb-200-3.3B

# Multilingual

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/rocsmt/test
INPUT_FILES[0]="${DATA_DIR[0]}/gpt.raw.en.test"
SRC_LANG[0]=eng_Latn
TGT_LANG[0]=fra_Latn

CORPUS[1]=rocsmt
DATA_DIR[1]=$DATASETS/rocsmt/test
INPUT_FILES[1]="${DATA_DIR[1]}/gpt.raw.en.test"
SRC_LANG[1]=eng_Latn
TGT_LANG[1]=deu_Latn

CORPUS[2]=rocsmt
DATA_DIR[2]=$DATASETS/rocsmt/test
INPUT_FILES[2]="${DATA_DIR[2]}/gpt.raw.en.test"
SRC_LANG[2]=eng_Latn
TGT_LANG[2]=ces_Latn

CORPUS[3]=rocsmt
DATA_DIR[3]=$DATASETS/rocsmt/test
INPUT_FILES[3]="${DATA_DIR[3]}/gpt.raw.en.test"
SRC_LANG[3]=eng_Latn
TGT_LANG[3]=ukr_Cyrl

CORPUS[4]=rocsmt
DATA_DIR[4]=$DATASETS/rocsmt/test
INPUT_FILES[4]="${DATA_DIR[4]}/gpt.raw.en.test"
SRC_LANG[4]=eng_Latn
TGT_LANG[4]=rus_Cyrl

# Non-standard English

CORPUS[13]=footweets
DATA_DIR[13]=$DATASETS/footweets
INPUT_FILES[13]="${DATA_DIR[13]}/gpt.detok.twitter.sent.en.txt"
SRC_LANG[13]=eng_Latn
TGT_LANG[13]=deu_Latn

CORPUS[14]=mtnt
DATA_DIR[14]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[14]="${DATA_DIR[14]}/gpt.en-fr.en"
SRC_LANG[14]=eng_Latn
TGT_LANG[14]=fra_Latn

CORPUS[15]=mtnt
DATA_DIR[15]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[15]="${DATA_DIR[15]}/gpt.en-ja.en"
SRC_LANG[15]=eng_Latn
TGT_LANG[15]=jpn_Jpan

# Non-standard French

CORPUS[16]=mmtc
DATA_DIR[16]=$DATASETS/mmtc
INPUT_FILES[16]="${DATA_DIR[16]}/gpt.test.fr-en.fr"
SRC_LANG[16]=fra_Latn
TGT_LANG[16]=eng_Latn

CORPUS[17]=mtnt
DATA_DIR[17]=$DATASETS/mtnt/MTNT2019
INPUT_FILES[17]="${DATA_DIR[17]}/gpt.fr-en.fr"
SRC_LANG[17]=fra_Latn
TGT_LANG[17]=eng_Latn

CORPUS[18]=pfsmb
DATA_DIR[18]=$DATASETS/pfsmb
INPUT_FILES[18]="${DATA_DIR[18]}/gpt.test.fr"
SRC_LANG[18]=fra_Latn
TGT_LANG[18]=eng_Latn

CORPUS[19]=foursquare
DATA_DIR[19]=$DATASETS/foursquare
INPUT_FILES[19]="${DATA_DIR[19]}/gpt.test.fr"
SRC_LANG[19]=fra_Latn
TGT_LANG[19]=eng_Latn


for i in {0..0}
do
    echo "Model: ${MODEL[$i]}"

    OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/outputs/${MODEL[$i]}/${CORPUS[$SLURM_ARRAY_TASK_ID]}
    OUTPUT_DIR=$OUTPUT_DIR_PREFIX/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}

    # Launch the generation of outputs
    for INPUT_FILE in ${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
    do 
        echo -e "\t - input file: $INPUT_FILE"

        python $HOME/robust-embeddings/src/sonar/generate-nllb.py \
            -i $INPUT_FILE \
            -s ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
            -t ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
            -o $OUTPUT_DIR \
            -m ${MODEL_NAME[$i]}
    done
done

echo "Done..."

