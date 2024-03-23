#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=ncm@a100
#SBATCH -C a100
##SBATCH --partition=gpu_p2
##SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=8               # total number of tasks across all nodes
#SBATCH --gres=gpu:8        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=8	# number of cores per task (8x3 = 24 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=1

echo "### Running $SLURM_JOB_NAME ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc 
source $HOME/.bash_profile
source $HOME/.bash_python_exports

set -e

EXPERIMENT_BASE_NUM="041"
LEARNING_RATE="0.0001" #--lr-scheduler inverse_sqrt"
CONFIG="distil:robertaugc-laserstd" #,distil:robertastd-laserstd"
MAX_EPOCH=100
DROPOUT=0.1

TEACHER_MODEL=$LASER/models/laser2.pt
STUDENT_MODEL=$WORK/models/roberta-base/pytorch_model.bin

EXPERIMENT_NUM[0]="${EXPERIMENT_BASE_NUM}"
PORT[0]="12341"
POOLING[0]="maxpool"
INIT[0]=""
PREPEND_BOS[0]=""

EXPERIMENT_NUM[1]="${EXPERIMENT_BASE_NUM}b"
PORT[1]="12342"
POOLING[1]="maxpool"
INIT[1]="--student-checkpoint-path $STUDENT_MODEL"
PREPEND_BOS[1]=""

EXPERIMENT_NUM[2]="${EXPERIMENT_BASE_NUM}c"
PORT[2]="12343"
POOLING[2]="meanpool"
INIT[2]=""
PREPEND_BOS[2]=""

EXPERIMENT_NUM[3]="${EXPERIMENT_BASE_NUM}d"
PORT[3]="12344"
POOLING[3]="meanpool"
INIT[3]="--student-checkpoint-path $STUDENT_MODEL"
PREPEND_BOS[3]=""

EXPERIMENT_NUM[4]="${EXPERIMENT_BASE_NUM}i"
PORT[4]="12345"
POOLING[4]="cls"
INIT[4]=""
PREPEND_BOS[4]="--prepend-bos"

EXPERIMENT_NUM[5]="${EXPERIMENT_BASE_NUM}j"
PORT[5]="12346"
POOLING[5]="cls"
INIT[5]="--student-checkpoint-path $STUDENT_MODEL"
PREPEND_BOS[5]="--prepend-bos"

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
    --dropout $DROPOUT \
    --activation-dropout 0.1 \
    --attention-dropout 0.1 \
    --max-tokens 4000 \
    --max-epoch $MAX_EPOCH \
    --student-bpe-symbol roberta \
    --left-pad-source True \
    --left-pad-target False \
    --layernorm-embedding \
    --encoder-embed-dim 768 \
    --encoder-layers 12 \
    --encoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 12 \
    --encoder-normalize-before \
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
    ${INIT[$SLURM_ARRAY_TASK_ID]}  ${PREPEND_BOS[$SLURM_ARRAY_TASK_ID]}

echo "Done..."
