#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node # IMPORTANT - only 1 task per dist per node!
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:20:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/mining/%x/%x_%j.log # Standard output and error log
##SBATCH --array=0
#SBATCH --exclusive
#SBATCH --qos=qos_gpu-dev #qos_gpu-t4

echo "### Running $SLURM_JOB_NAME ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

#module load cpuarch/amd pytorch-gpu/py3/2.0.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9
module load pytorch-gpu/py3/2.0.0

set -e

# Setting the experiment parameters

EXPERIMENT_NUM="050"
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/mining/experiment_$EXPERIMENT_NUM

SEED=42
ACCUMULATION_STEPS=1 
LEARNING_RATE=2e-5 # from Reimers and Gurevych (2020)
LR_SCHEDULER=constant_with_warmup

echo "Setting the accelerate config..."

if [ $SLURM_NNODES -eq 1 ]; then
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_single_node_config.yaml
else
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_multi_node_config.yaml
fi

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
GPUS_PER_NODE=8
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
    --mixed_precision no \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
    $HOME/robust-embeddings/src/mining/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --learning-rate $LEARNING_RATE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-scheduler-type $LR_SCHEDULER \
    --dataloader-workers $SLURM_CPUS_PER_TASK
    "

export CMD="$LAUNCHER $PROGRAM"

srun bash -c "$CMD" 

echo "Done..."

