#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=64	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=4:00:00                # Time limit hrs:min:sec
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/sonar/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

#module load anaconda-py3/2023.09 pytorch-gpu/py3/2.3.0
module load cpuarch/amd pytorch-gpu/py3/2.3.0 libsndfile/1.0.28 intel-oneapi-tbb/2021.9

set -e

EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047
SEED=42

python $HOME/robust-embeddings/src/sonar/draft.py 

echo "Done..."

