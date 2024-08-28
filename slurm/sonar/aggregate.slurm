#!/bin/bash

#SBATCH --job-name=aggregate      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-5

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc
source $HOME/.bash_profile

BASE_EXPERIMENT_NUM=047
CURRENT_EXPERIMENT_NUM=${BASE_EXPERIMENT_NUM}
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_$CURRENT_EXPERIMENT_NUM

MODELS=(nllb1b sonar nllb600m) # rosonar)

# Multilingual
TABLE_NAME[0]=multilingual
CORPORA[0]="rocsmt flores200"
LANG_PAIRS[0]="eng_Latn-ces_Latn eng_Latn-deu_Latn eng_Latn-fra_Latn eng_Latn-rus_Cyrl eng_Latn-ukr_Cyrl"

# Standard bilingual
TABLE_NAME[1]=bilingual
CORPORA[1]="wmt2015 flores200"
LANG_PAIRS[1]="eng_Latn-fra_Latn fra_Latn-eng_Latn"

# Non-standard English
TABLE_NAME[2]=ugc_english
CORPORA[2]="footweets mtnt"
LANG_PAIRS[2]="eng_Latn-deu_Latn eng_Latn-fra_Latn eng_Latn-jpn_Jpan"

# Non-standard French
TABLE_NAME[3]=ugc_french
CORPORA[3]="foursquare mmtc mtnt pfsmb"
LANG_PAIRS[3]="fra_Latn-eng_Latn"

# Autoencode
TABLE_NAME[4]=autoencode
CORPORA[4]="rocsmt"
LANG_PAIRS[4]="fra_Latn eng_Latn eng_Latn-eng_Latn"

# Artificial
TABLE_NAME[5]=artificial
CORPORA[5]="rocsmt_artificial"
LANG_PAIRS[5]="eng_Latn-ces_Latn eng_Latn-deu_Latn eng_Latn-fra_Latn eng_Latn-rus_Cyrl eng_Latn-ukr_Cyrl"
SEEDS="1000 1001 1002 1003 1004"
PROBAS="0.1 0.2 0.3 0.4 0.5"

# Copy baselines outputs

for MODEL in ${MODELS[@]:0:3}
do
    if [ ! -d $EXPERIMENT_DIR/outputs/$MODEL ]
    then
        cp -r $EXPERIMENT_DIR/../experiment_$BASE_EXPERIMENT_NUM/outputs/$MODEL $EXPERIMENT_DIR/outputs/
    fi
done

# Aggregate scores

if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]
then
    python $HOME/robust-embeddings/src/sonar/aggregate-artificial.py \
        -i $EXPERIMENT_DIR \
        -t ${TABLE_NAME[$SLURM_ARRAY_TASK_ID]} \
        -c ${CORPORA[$SLURM_ARRAY_TASK_ID]} \
        -l ${LANG_PAIRS[$SLURM_ARRAY_TASK_ID]} \
        -m $MODELS \
        -s $SEEDS \
        -p $PROBAS
else
    python $HOME/robust-embeddings/src/sonar/aggregate.py \
        -i $EXPERIMENT_DIR \
        -t ${TABLE_NAME[$SLURM_ARRAY_TASK_ID]} \
        -c ${CORPORA[$SLURM_ARRAY_TASK_ID]} \
        -l ${LANG_PAIRS[$SLURM_ARRAY_TASK_ID]} \
        -m $MODELS
fi

echo "Done..."
