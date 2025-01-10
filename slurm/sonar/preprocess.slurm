#!/bin/bash

#SBATCH --job-name=preprocess      # Job name
#SBATCH --account=rnh@v100
#SBATCH	-C v100-32g
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-999

echo "### Running $SLURM_JOB_NAME ###"

source $HOME/.bashrc 
source $HOME/.bash_profile

# Use pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

python $HOME/robust-embeddings/src/sonar/preprocess.py --split train --shard $SLURM_ARRAY_TASK_ID --num-processes $SLURM_CPUS_PER_TASK

if [ $SLURM_ARRAY_TASK_ID -lt 32 ]; then
    python $HOME/robust-embeddings/src/sonar/preprocess.py --split valid --shard $SLURM_ARRAY_TASK_ID --num-processes $SLURM_CPUS_PER_TASK
fi

echo "Done..."