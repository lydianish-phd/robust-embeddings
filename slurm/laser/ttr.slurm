#!/bin/bash

#SBATCH --job-name=ttr      # Job name
#SBATCH --account=rnh@cpu
#SBATCH --nodes=1		# node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=40	# number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=1:00:00
#SBATCH --output=/gpfsscratch/rech/rnh/udc54vm/logs/robust-embeddings/laser/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0,10,20,30,40,50
set -e 

echo "### Running $SLURM_JOB_NAME ###"

module purge
module load pytorch-gpu/py3/1.13.0

source $HOME/.bashrc
source $HOME/.bash_profile
source $HOME/.bash_python_exports

SRC_DIR=$HOME/robust-embeddings/src/laser

python $SRC_DIR/ttr_flores.py -i $DATASETS/flores200/tok
python $SRC_DIR/ttr_natural.py -i $DATASETS/multilexnorm2021/en/tok/ -c "multilexnorm2021"
python $SRC_DIR/ttr_natural.py -i $DATASETS/rocsmt/tok/ -c "rocsmt"
# python $SRC_DIR/ttr_natural.py -i $DATASETS/oscar/mini/4Mlrec/tok/robertaugc-laserstd/

echo "Done..."
