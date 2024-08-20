#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=8               # number of tasks per node
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=8	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-13%1

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

# Use pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

# Set the experiment parameters
EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047c
SEED=42
CHECKPOINT_DIR=$EXPERIMENT_DIR/models
LR_SCHEDULER=constant #_with_warmup
FRACTION_EN_FR=5
FRACTION_FR=2
FRACTION_EN=1

# Set the IP and port for the main process
SRC_DIR=$HOME/robust-embeddings/src/sonar
source $SRC_DIR/main-process-ip-and-port.sh
echo "Main process IP: $MASTER_ADDRESS and port: $MASTER_PORT"

# Launch the training
accelerate launch $SRC_DIR/train.py \
    -o $EXPERIMENT_DIR \
    --seed $SEED \
    --checkpoint-dir $CHECKPOINT_DIR \
    --lr-scheduler-type $LR_SCHEDULER \
    --fraction-en-fr $FRACTION_EN_FR \
    --fraction-fr $FRACTION_FR \
    --fraction-en $FRACTION_EN

echo "Done..."

