#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=2		# node count
#SBATCH --ntasks-per-node=8               # number of tasks per node
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0 #-13%1

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

# Using pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047d
SEED=42
CHECKPOINT_DIR=$EXPERIMENT_DIR/models

# Setting the main process IP and port
ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/default_config.yaml
output=$(python -c "import idr_torch; print(f'{idr_torch.master_addr},{idr_torch.master_port}')")
MASTER_ADDRESS=$(echo $output | cut -d, -f1)
MASTER_PORT=$(echo $output | cut -d, -f2)
sed -i "s/^main_process_ip:.*/main_process_ip: $MASTER_ADDRESS/" $ACCELERATE_CONFIG
sed -i "s/^main_process_port:.*/main_process_port: $MASTER_PORT/" $ACCELERATE_CONFIG

accelerate launch $HOME/robust-embeddings/src/sonar/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR

echo "Done..."

