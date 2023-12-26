export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PATH=/usr/local/cuda/bin:$PATH

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9907

llama_path=pretrained_models/llama2-7b-hf
whisper_path=pretrained_models/whisper-small

DATA_ROOT=data/multitask_20231211_2
# DATA_ROOT=data/multitask/AAC 
SAVE_ROOT=checkpoints/20231211-traintag_newtag_4
# SAVE_ROOT=debug
mkdir -p $SAVE_ROOT
torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    blsp/train_multitask.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "*.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 24 \
    --num_train_epochs 50 \
    \
    --llama_model $llama_path \
    --whisper_model $whisper_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --log_on_each_node False \
