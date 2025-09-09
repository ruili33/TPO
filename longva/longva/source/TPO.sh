#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

n_nodes=1
bs=1
# OUTPUT of stage 2 script
STAGE2_PATH=lmms-lab/LongVA-7B
# Final output checkpoint path
OUTPUT=output_dir
mkdir -p $OUTPUT
# --data_mixture hound+llava15+vflan \

torchrun --nnodes=$n_node --nproc_per_node=1 --master_port=25005 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    longva/longva/train/train_mem_dpo.py \
    --model_name_or_path $STAGE2_PATH \
    --data_path <data_path> \
    --image_folder <image_folder> \
    --deepspeed ./longva/longva/source/zero2.json \
    --rpo_alpha 0.5 --beta 0.3 --gamma 0 \
    --version qwen \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_feature patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --bf16 true \
    --output_dir $OUTPUT/checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --learning_rate 4e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb \
    --num_video_frames 32\
    --p_bar 0\
    --scale 0 \
    --lora_enable False
   