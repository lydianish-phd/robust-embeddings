#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=ncm@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=2		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=1:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0 #-10%1

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
ACCUMULATION_STEPS=32
LEARNING_RATE=5e-4
LR_SCHEDULER=inverse_sqrt
NO_UGC_EN="--no-ugc-en"

echo "Setting the accelerate config..."

if [ $SLURM_NNODES -eq 1 ]; then
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_single_node_config.yaml
else
    DEFAULT_ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/accelerate_multi_node_config.yaml
fi

ACCELERATE_CONFIG=$EXPERIMENT_DIR/accelerate_config.yaml
cp $DEFAULT_ACCELERATE_CONFIG $ACCELERATE_CONFIG

output=$(srun python -c "import idr_torch; print(f'{idr_torch.master_addr},{idr_torch.master_port}')")
MASTER_ADDRESS=$(echo $output | cut -d, -f1)
MASTER_PORT=$(echo $output | cut -d, -f2 | cut -d' ' -f1)

sed -i "s/^num_machines:.*/num_machines: $SLURM_NNODES/" $ACCELERATE_CONFIG
sed -i "s/^num_processes:.*/num_processes: $SLURM_NPROCS/" $ACCELERATE_CONFIG
sed -i "s/^main_process_ip:.*/main_process_ip: $MASTER_ADDRESS/" $ACCELERATE_CONFIG
sed -i "s/^main_process_port:.*/main_process_port: $MASTER_PORT/" $ACCELERATE_CONFIG

echo "Launch the training the training script..."

srun accelerate launch --config_file $ACCELERATE_CONFIG $HOME/robust-embeddings/src/sonar/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --learning-rate $LEARNING_RATE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-scheduler-type $LR_SCHEDULER \
    $NO_UGC_EN

echo "Done..."

