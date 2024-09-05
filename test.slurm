#!/bin/bash

#SBATCH -J embed
#SBATCH -A kytam
#SBATCH -w hhnode-ib-234
#SBATCH --nodes=1 
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -p dbm
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tangyixuanemail@gmail.com 
#SBATCH --gres=gpu:a800sxm80gb:8


echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

set -x -e
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export LOGLEVEL=INFO
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NNODES=$SLURM_JOB_NUM_NODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# handle timeouts
export NCCL_IB_TIMEOUT=20

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
### init virtual environment 
### init virtual environment 
module load anaconda3
###conda remove -n py10 --all

###conda create -n py10 python=3.10
source activate unsloth_env

export PYTHONUNBUFFERED=TRUE
python run_mteb.py

conda deactivate