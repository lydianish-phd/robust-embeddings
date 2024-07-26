#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=12:30:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
MODELS="nllb1b sonar nllb600m"

CORPUS=rocsmt_artificial
DATA_DIR=$DATASETS/rocsmt
SRC_DIR=$DATA_DIR/artificial
REF_DIR=$DATA_DIR/test

REF_FILE[0]=$REF_DIR/ref.fr.test
LANG_PAIR[0]=eng_Latn-fra_Latn

REF_FILE[1]=$REF_DIR/ref.de.test
LANG_PAIR[1]=eng_Latn-deu_Latn

REF_FILE[2]=$REF_DIR/ref.cs.test
LANG_PAIR[2]=eng_Latn-ces_Latn

REF_FILE[3]=$REF_DIR/ref.uk.test
LANG_PAIR[3]=eng_Latn-ukr_Cyrl

REF_FILE[4]=$REF_DIR/ref.ru.test
LANG_PAIR[4]=eng_Latn-rus_Cyrl

PROBAS="0.1 0.2 0.3 0.4 0.5"

# Launch the evaluation of outputs

for MODEL in $MODELS
do
    echo "Model: $MODEL"

    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL/$CORPUS/${LANG_PAIR[$SLURM_ARRAY_TASK_ID]}
    
    for SEED in {1000..1004}
    do
        echo -e "\t - seed $SEED"

        for PROBA in $PROBAS
        do
            echo -e "\t\t - proba $PROBA"

            SRC_FILE=$SRC_DIR/$SEED/$PROBA/ugc/norm.en_mix_all.test
            SYS_FILE=$OUTPUT_DIR/$SEED/$PROBA/norm.en_mix_all.test.out
        
            python $HOME/robust-embeddings/src/sonar/evaluate-translation.py \
                --src-file $SRC_FILE \
                --sys-file $SYS_FILE \
                --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]}
        done
    done 
done

echo "Done..."

