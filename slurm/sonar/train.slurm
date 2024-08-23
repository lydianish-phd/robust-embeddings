#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=2		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:2        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0 #-10%1
#SBATCH --qos=qos_gpu-dev

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

# Use pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

# Set the experiment parameters

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047d
mkdir -p $EXPERIMENT_DIR
SEED=42
CHECKPOINT_DIR=$EXPERIMENT_DIR/models
ACCUMULATION_STEPS=$((32 / SLURM_NNODES))
LEARNING_RATE=5e-4
LR_SCHEDULER=inverse_sqrt
NO_UGC_EN="--no-ugc-en"

echo "Setting the accelerate config..."

if [ $SLURM_NNODES -eq 1 ]; then
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_single_node_config.yaml
else
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_multi_node_config.yaml
fi

output=$(srun python -c "import idr_torch; print(f'{idr_torch.master_addr},{idr_torch.master_port},{idr_torch.local_rank}')")
MASTER_ADDR=$(echo $output | cut -d, -f1)
MASTER_PORT=$(echo $output | cut -d, -f2 | cut -d' ' -f1)
MACHINE_RANK=$(echo $output | cut -d, -f3 | cut -d' ' -f1)


GPUS_PER_NODE=2
NUM_PROCESSES=$(expr $SLURM_NNODES \* $GPUS_PER_NODE)

N_GPUS=$(srun nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

echo "output: $output"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MACHINE_RANK: $MACHINE_RANK"
echo "ACCUMULATION_STEPS: $ACCUMULATION_STEPS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "N_GPUS: $N_GPUS"

echo "Launching the training script..."

accelerate launch \
    --config_file $DEFAULT_ACCELERATE_CONFIG \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $MACHINE_RANK \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    $HOME/robust-embeddings/src/sonar/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --learning-rate $LEARNING_RATE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-scheduler-type $LR_SCHEDULER \
    $NO_UGC_EN

echo "Done..."

