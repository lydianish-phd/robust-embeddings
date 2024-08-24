#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=4		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node # IMPORTANT - only 1 task per dist per node!
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-3%1
##SBATCH --qos=qos_gpu-dev

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

# Use pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

# Setting the experiment parameters
EXPERIMENT_NUM="047d"
NO_UGC_EN="" #"--no-ugc-en"

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_$EXPERIMENT_NUM
mkdir -p $EXPERIMENT_DIR
CHECKPOINT_DIR=$EXPERIMENT_DIR/models

SEED=42
ACCUMULATION_STEPS=$((32 / SLURM_NNODES))
LEARNING_RATE=5e-4
LR_SCHEDULER=inverse_sqrt

echo "Setting the accelerate config..."

if [ $SLURM_NNODES -eq 1 ]; then
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_single_node_config.yaml
else
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_multi_node_config.yaml
fi

output=$(srun python -c "import idr_torch; print(f'{idr_torch.master_addr},{idr_torch.master_port}')")
MASTER_ADDR=$(echo $output | cut -d, -f1)
MASTER_PORT=$(echo $output | cut -d, -f2 | cut -d' ' -f1)
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
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
    $HOME/robust-embeddings/src/sonar/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --learning-rate $LEARNING_RATE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-scheduler-type $LR_SCHEDULER \
    $NO_UGC_EN
    "

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 

echo "Done..."

