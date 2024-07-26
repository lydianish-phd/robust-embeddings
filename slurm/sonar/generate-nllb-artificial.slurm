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
#SBATCH --time=5:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047

MODEL[0]=nllb600m
MODEL_NAME[0]=facebook/nllb-200-distilled-600m

MODEL[1]=nllb1b
MODEL_NAME[1]=facebook/nllb-200-distilled-1.3B

CORPUS=rocsmt_artificial
INPUT_DIR=$DATASETS/rocsmt/artificial

SRC_LANG[0]=eng_Latn
TGT_LANG[0]=fra_Latn

SRC_LANG[1]=eng_Latn
TGT_LANG[1]=deu_Latn

SRC_LANG[2]=eng_Latn
TGT_LANG[2]=ces_Latn

SRC_LANG[3]=eng_Latn
TGT_LANG[3]=ukr_Cyrl

SRC_LANG[4]=eng_Latn
TGT_LANG[4]=rus_Cyrl

PROBAS="0.1 0.2 0.3 0.4 0.5"

# Launch the generation of outputs

for i in {0..1}
do
    echo "Model: ${MODEL[$i]}"

    OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/outputs/${MODEL[$i]}/$CORPUS/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}

    for SEED in {1000..1004}
    do
        echo "\t - seed $SEED"

        for PROBA in $PROBAS
        do
            echo -e "\t\t - proba $PROBA"
            
            INPUT_FILE=$INPUT_DIR/$SEED/$PROBA/ugc/norm.en_mix_all.test
            OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$SEED/$PROBA

            python $HOME/robust-embeddings/src/sonar/generate-nllb.py \
                -i $INPUT_FILE \
                -s ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
                -t ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
                -o $OUTPUT_DIR -m ${MODEL_NAME[$i]}
        done
    done
done

echo "Done..."

