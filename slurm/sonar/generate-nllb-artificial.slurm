#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2
##SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=32	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using anaconda-py3/2023.09 which has pytorch 2.2.2 but NO accelerate (otherwise fails to load NLLB pretrained model from HF)
# module load arch/a100 anaconda-py3/2023.09 libsndfile/1.0.28 intel-oneapi-tbb/2021.9
module load pytorch-gpu/py3/2.0.0 #V100

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047r

# Base models

MODEL[0]=nllb600m
MODEL_NAME[0]=facebook/nllb-200-distilled-600m
MODEL_PATH[0]=facebook/nllb-200-distilled-600m

MODEL[1]=nllb1b
MODEL_NAME[1]=facebook/nllb-200-distilled-1.3B
MODEL_PATH[1]=facebook/nllb-200-distilled-1.3B

MODEL[2]=nllb3b
MODEL_NAME[2]=facebook/nllb-200-3.3B
MODEL_PATH[2]=facebook/nllb-200-3.3B

# Fine-tuned models

MODEL[3]=nllb600m_ft_nat_en_fr
MODEL_NAME[3]=facebook/nllb-200-distilled-600m
MODEL_PATH[3]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/natural/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[4]=nllb600m_ft_nat_fr_en
MODEL_NAME[4]=facebook/nllb-200-distilled-600m
MODEL_PATH[4]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/natural/fra_Latn-eng_Latn/nllb600m/checkpoint_best

MODEL[5]=nllb600m_ft_syn_en_fr
MODEL_NAME[5]=facebook/nllb-200-distilled-600m
MODEL_PATH[5]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/synthetic/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[6]=nllb600m_ft_nat_en_fr_bal
MODEL_NAME[6]=facebook/nllb-200-distilled-600m
MODEL_PATH[6]=$EXPERIMENTS/finetuning-ugc/experiment_052m/models/natural/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[7]=nllb600m_ft_nat_fr_en_bal
MODEL_NAME[7]=facebook/nllb-200-distilled-600m
MODEL_PATH[7]=$EXPERIMENTS/finetuning-ugc/experiment_052m/models/natural/fra_Latn-eng_Latn/nllb600m/checkpoint_best

MODEL[8]=nllb600m_ft_syn_en_fr_bal
MODEL_NAME[8]=facebook/nllb-200-distilled-600m
MODEL_PATH[8]=$EXPERIMENTS/finetuning-ugc/experiment_052m/models/synthetic/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[9]=nllb600m_ft_syn_plus_en_fr
MODEL_NAME[9]=facebook/nllb-200-distilled-600m
MODEL_PATH[9]=$EXPERIMENTS/finetuning-ugc/experiment_052k/models/synthetic_plus/eng_Latn-fra_Latn/nllb600m/checkpoint_best

MODEL[10]=nllb600m_ft_syn_plus_en_fr_bal
MODEL_NAME[10]=facebook/nllb-200-distilled-600m
MODEL_PATH[10]=$EXPERIMENTS/finetuning-ugc/experiment_052m/models/synthetic_plus/eng_Latn-fra_Latn/nllb600m/checkpoint_best


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

for i in {3..5}
do
    echo "Model: ${MODEL[$i]}"

    OUTPUT_DIR_PREFIX=$EXPERIMENT_DIR/outputs/${MODEL[$i]}/$CORPUS/${SRC_LANG[$SLURM_ARRAY_TASK_ID]}-${TGT_LANG[$SLURM_ARRAY_TASK_ID]}

    for SEED in {1000..1004}
    do
        echo -e "\t - seed $SEED"

        for PROBA in $PROBAS
        do
            echo -e "\t\t - proba $PROBA"
            
            INPUT_FILE=$INPUT_DIR/$SEED/$PROBA/ugc/test/norm.en_mix_all.test
            OUTPUT_DIR=$OUTPUT_DIR_PREFIX/$SEED/$PROBA

            python $HOME/robust-embeddings/src/sonar/generate-nllb.py \
                -i $INPUT_FILE \
                -s ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
                -t ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
                -o $OUTPUT_DIR \
                -m ${MODEL_NAME[$i]} \
                -p ${MODEL_PATH[$i]}
        done
    done
done

echo "Done..."

