#!/bin/bash

#SBATCH --job-name=preprocess      # Job name
#SBATCH --account=rnh@v100
#SBATCH	-C v100-32g
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=0:20:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/mining/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

source $HOME/.bashrc 
source $HOME/.bash_profile

module load pytorch-gpu/py3/2.0.0

set -e

NUM_PROC=1 #$((SLURM_CPUS_PER_TASK / 2))

python $HOME/robust-embeddings/src/mining/preprocess_shard.py --num-processes $NUM_PROC

echo "Done..."