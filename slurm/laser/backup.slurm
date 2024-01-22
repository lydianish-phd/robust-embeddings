#!/bin/bash

#SBATCH --job-name=backup      # Job name
#SBATCH --account=rnh@v100
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=20:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME ###"

EXPERIMENT_NUM=029
EXPERIMENT_DIR=experiments/robust-embeddings/laser

echo "Backing up files..."

tar -czvf $STORE/$EXPERIMENT_DIR/experiment_$EXPERIMENT_NUM.tar.gz $SCRATCH/$EXPERIMENT_DIR/experiment_$EXPERIMENT_NUM*

echo "Done..."
