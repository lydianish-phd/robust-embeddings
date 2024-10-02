#!/bin/bash

#SBATCH --job-name=preprocess      # Job name
#SBATCH --account=rnh@v100
#SBATCH	-C v100-32g
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=6:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

source $HOME/.bashrc 
source $HOME/.bash_profile

# Use pytorch-gpu/py3/2.3.0 which has pytorch 2.2.2 and accelerate
module load arch/a100 pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

python $HOME/robust-embeddings/src/sonar/preprocess.py --num-processes $SLURM_CPUS_PER_TASK

echo "Done..."