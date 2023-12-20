#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=2		# node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
#SBATCH --gres=gpu:4        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=10	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=3

echo "### Running $SLURM_JOB_NAME ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc 
source $HOME/.bash_profile
source $HOME/.bash_python_exports

set -e

EXPERIMENT_BASE_NUM="032"
LEARNING_RATE="0.00005" #--lr-scheduler inverse_sqrt"
CONFIG="distil:robertaugc-laserstd" #,distil:robertastd-laserstd"

TEACHER_MODEL=$LASER/models/laser2.pt
STUDENT_MODEL=$WORK/models/roberta-base/pytorch_model.bin

EXPERIMENT_NUM[0]="${EXPERIMENT_BASE_NUM}"
PORT[0]="12341"
POOLING[0]="maxpool"
INIT[0]=""

EXPERIMENT_NUM[1]="${EXPERIMENT_BASE_NUM}b"
PORT[1]="12342"
POOLING[1]="maxpool"
INIT[1]="--student-checkpoint-path $STUDENT_MODEL"

EXPERIMENT_NUM[2]="${EXPERIMENT_BASE_NUM}c"
PORT[2]="12343"
POOLING[2]="meanpool"
INIT[2]=""

EXPERIMENT_NUM[3]="${EXPERIMENT_BASE_NUM}d"
PORT[3]="12344"
POOLING[3]="meanpool"
INIT[3]="--student-checkpoint-path $STUDENT_MODEL"


EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_${EXPERIMENT_NUM[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH=$EXPERIMENT_DIR/train-config.json

OUTPUT_DIR=$EXPERIMENT_DIR/models
TENSORBOARD_DIR=$EXPERIMENT_DIR/tensorboard

echo "Training student model $SLURM_ARRAY_TASK_ID..."

srun fairseq-train \
    $CONFIG_PATH \
    --user-dir $FAIRSEQ_PATH/examples/nllb/laser_distillation \
    --log-interval 100  \
    --log-format simple \
    --task laser_distillation \
    --arch laser_transformer \
    --sentemb-criterion ${POOLING[$SLURM_ARRAY_TASK_ID]} \
    --prepend-bos  \
    --criterion encoder_similarity \
    --mse \
    --save-dir $OUTPUT_DIR \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6 \
    --lr $LEARNING_RATE \
    --warmup-updates 1000 \
    --clip-norm 5 \
    --update-freq 1 \
    --dropout 0.3 \
    --max-tokens 4000 \
    --max-epoch 100 \
    --student-bpe-symbol roberta \
    --left-pad-source False \
    --left-pad-target True \
    --encoder-embed-dim 768 \
    --encoder-layers 12 \
    --encoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 12 \
    --decoder-embed-dim 1 \
    --decoder-layers 1 \
    --decoder-embed-dim 1 \
    --decoder-ffn-embed-dim 1 \
    --decoder-attention-heads 1 \
    --decoder-lang-embed-dim 1 \
    --ddp-backend=no_c10d \
    --save-interval-updates 30000 \
    --disable-validation \
    --teacher-checkpoint-path $TEACHER_MODEL \
    --lambda-self 0 \
    --lambda-mask 0 \
    --lambda-distil 1.0 \
    --student-teacher-config $CONFIG \
    --tensorboard-logdir $TENSORBOARD_DIR \
    --distributed-port ${PORT[$SLURM_ARRAY_TASK_ID]} \
    #${INIT[$SLURM_ARRAY_TASK_ID]}

echo "Done..."
