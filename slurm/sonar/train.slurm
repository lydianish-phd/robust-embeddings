#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@h100
#SBATCH -C h100
#SBATCH --nodes=4		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node # IMPORTANT - only 1 task per dist per node!
#SBATCH --gres=gpu:4        # numper of GPUs per node
#SBATCH --cpus-per-task=96	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=100:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --exclusive
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --array=0-1

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

# Use pytorch-gpu/py3/2.3.1 which has pytorch 2.2.2 and accelerate
module load arch/h100 pytorch-gpu/py3/2.3.1 libsndfile/1.0.28 intel-oneapi-mkl/2024.1

source $HOME/.bashrc
source $HOME/.bash_profile

set -e

DATA_CONFIG=$SLURM_ARRAY_TASK_ID

MODEL[0]="rosonar"
UGC_EN[0]=True

MODEL[1]="rosonar_std"
UGC_EN[1]=False

# Setting the experiment parameters

EXPERIMENT_NUM="047r"
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_$EXPERIMENT_NUM

SEED=42
ACCUMULATION_STEPS=$((64 / SLURM_NNODES))
LEARNING_RATE=7e-3
LR_SCHEDULER=inverse_sqrt
MAX_STEPS=1000000

echo "Setting the accelerate config..."

if [ $SLURM_NNODES -eq 1 ]; then
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_single_node_config.yaml
else
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_multi_node_config.yaml
fi

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((12345+DATA_CONFIG))
GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MACHINE_RANK: $SLURM_PROCID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "NUM_PROCESSES: $NUM_PROCESSES"

echo "Launching the training script..."

export LAUNCHER="accelerate launch \
    --config_file $DEFAULT_ACCELERATE_CONFIG \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --mixed_precision bf16 \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
    $HOME/robust-embeddings/src/sonar/train.py \
    -o $EXPERIMENT_DIR \
    --model-name ${MODEL[$DATA_CONFIG]} \
    --seed $SEED \
    --learning-rate $LEARNING_RATE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-scheduler-type $LR_SCHEDULER \
    --max-steps $MAX_STEPS \
    --ugc-en ${UGC_EN[$DATA_CONFIG]} \
    --dataloader-workers $SLURM_CPUS_PER_TASK \
    "

export CMD="$LAUNCHER $PROGRAM"

srun bash -c "$CMD" 

echo "Done..."

