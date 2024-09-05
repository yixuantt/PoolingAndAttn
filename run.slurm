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
module load anaconda3
source activate unsloth_env
pip install transformers==4.40.2
export PYTHONUNBUFFERED=TRUE

export LAUNCHER="accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
export PROGRAM="model5_trainer.py \
    --dataset_name yixuantt/ir_data \
    --max_seq_length 512 \
    --cache_dir /scratch/PI/sbm/yixuan/ \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --bf16 \
    --max_steps 1000 \
    --save_steps 100 \
    --save_total_limit 15 \
    --logging_steps 1 \
    --weight_decay 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps  256 \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --output_dir /scratch/PI/sbm/yixuan/test/ "

export CMD="$LAUNCHER $PROGRAM"
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
echo "END TIME: $(date)"


conda deactivate