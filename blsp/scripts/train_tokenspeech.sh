export NCCL_DEBUG=INFO
# export NCCL_LAUNCH_MODE=PARALLEL
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=0
export PATH=/usr/local/cuda/bin:$PATH
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

llama_path=pretrained_models/llama2-hf-7b

DATA_ROOT=data/speech_unit/w2v_8196_nomerge
DATA_ROOT=data/speech_unit/20240612_emo_caption

SAVE_ROOT=checkpoints/model_librispeech_960
SAVE_ROOT=checkpoints/hubert
SAVE_ROOT=checkpoints/gpu-kmeans
SAVE_ROOT=checkpoints/gpu-kmeans_nobpe_2
SAVE_ROOT=checkpoints/partial-hunbert-2
SAVE_ROOT=checkpoints/conversion_speech_hubert
SAVE_ROOT=checkpoints/conversion_speech_hubert_new
SAVE_ROOT=checkpoints/conversion_speech_hubert_new_chinese
SAVE_ROOT=checkpoints/conversion_speech_hubert_new_chinese_asr_sft_new
SAVE_ROOT=checkpoints/speeech_hubert_new_emo_cap_3
# SAVE_ROOT=checkpoints/asr_w2vbert_8196

mkdir -p $SAVE_ROOT

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    blsp/train_tokenspeech.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "all_shuf.jsonl" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_steps 2000 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 40 \
    \
    --llama_model $llama_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 1 \
    --overwrite_output_dir 
    # --resume_from_checkpoint checkpoints/speeech_hubert_new_emo_cap/checkpoint-1600
    # --log_on_each_node False \