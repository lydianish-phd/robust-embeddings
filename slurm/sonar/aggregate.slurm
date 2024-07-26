#!/bin/bash

#SBATCH --job-name=aggregate      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc
source $HOME/.bash_profile

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047

MODELS="sonar nllb600m nllb1b"

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
CORPORA[3]="mmtc mtnt pfsmb"
LANG_PAIRS[3]="fra_Latn-eng_Latn"

# Autoencode
TABLE_NAME[4]=autoencode
CORPORA[4]="rocsmt"
LANG_PAIRS[4]="fra_Latn eng_Latn eng_Latn-eng_Latn"

# Aggregate scores

python $HOME/robust-embeddings/src/sonar/aggregate.py \
    -i $EXPERIMENT_DIR \
    -t ${TABLE_NAME[$SLURM_ARRAY_TASK_ID]} \
    -c ${CORPORA[$SLURM_ARRAY_TASK_ID]} \
    -l ${LANG_PAIRS[$SLURM_ARRAY_TASK_ID]} \
    -m $MODELS

echo "Done..."
