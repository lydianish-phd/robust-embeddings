#!/bin/bash

#SBATCH --job-name=train      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node # IMPORTANT - only 1 task per dist per node!
#SBATCH --gres=gpu:8        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=nomultithread       # we get physical cores not logical
#SBATCH --time=100:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/robust-embeddings/mining/%x/%x_%j.log # Standard output and error log
#SBATCH --exclusive
#SBATCH --qos=qos_gpu-t4

echo "### Running $SLURM_JOB_NAME ###"

module purge

source $HOME/.bashrc 
source $HOME/.bash_profile

module load pytorch-gpu/py3/1.13.0

set -e

BASE_DIR=$DATASETS/rosonar/monolingual #community-oscar/data/2024-33
DATA_DIR=$BASE_DIR/cleaned-mtnt #cleaned
OUTPUT_DIR=$BASE_DIR/mined-mtnt #mined

SRC_LANG=eng
TGT_LANG=fra

MODEL_DIR=$SCRATCH/LASER/models
VOCAB_DIR=$SCRATCH/LASER/models

MAX_LINES_PER_SHARD=1000 #8000000
TMP_DIR=/scratch/lnishimw/tmp/slurm
CPUS_PER_TASK=24
CONSTRAINT=""
PARTITION=$SLURM_JOB_PARTITION

export HYDRA_FULL_ERROR=1

cd $WORK/stopes/stopes/pipelines/bitext/

python global_mining_pipeline.py \
    output_dir=$OUTPUT_DIR \
    src_lang=$SRC_LANG tgt_lang=$TGT_LANG \
    +data=oscar_cc \
    data.data_shard_dir=$DATA_DIR \
    embed_text=laser2 \
    model_dir=$MODEL_DIR \
    vocab_dir=$VOCAB_DIR \
    local_tmp_dir=$TMP_DIR \
    max_shard_size=$MAX_LINES_PER_SHARD \
    train_index.config.num_cpu=$CPUS_PER_TASK \
    train_index.config.constraint=$CONSTRAINT \
    populate_index.config.num_cpu=$CPUS_PER_TASK \
    calculate_distances.config.constraint=$CONSTRAINT \
    mine_indexes.config.num_cpu=$CPUS_PER_TASK \
    mine_sentences.config.num_cpu=$CPUS_PER_TASK \
    launcher.partition=$PARTITION 

echo "Done..."
