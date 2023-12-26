export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


llama_path=pretrained_models/llama2-7b-hf

DATA_ROOT=/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/data/stage1
SAVE_ROOT=/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/stage1

mkdir -p $SAVE_ROOT

python -m torch.distributed.run --nproc_per_node=4 blsp/train_stage1.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "train_alpaca.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    \
    --llama_model $llama_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 1