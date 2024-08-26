#!/bin/bash

#SBATCH --job-name=plot      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=1:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log

set -e 

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load cpuarch/amd pytorch-gpu/py3/1.10.0-AMD

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

OUTPUT_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_032_eval

python $HOME/robust-embeddings/src/laser/plot_distances.py -d $OUTPUT_EXPERIMENT_DIR

echo "Done..."
