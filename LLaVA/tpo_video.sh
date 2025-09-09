export ADDR=${master_addr:-"127.0.0.1"}
export RANK=${SLURM_PROCID:-"0"}

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# TPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="lmms-lab/LLaVA-Video-7B-Qwen2"
EPOCH=1
beta=0.2

TPO_RUN_NAME="llava-video-qwen2-7b_tpo-beta${beta}-epoch${EPOCH}"
TPO_CLEAN_NAME="${TPO_RUN_NAME##*/}"
OUTPUT_DIR="output_dir/${TPO_CLEAN_NAME}"
DATA_PATH="your_data_path"

echo $TPO_RUN_NAME

torchrun --nproc_per_node=1 --nnodes=1 --node_rank="${RANK}" --master_addr="${ADDR}" --master_port=25005 \
    llava/train/train_dpo.py \
    --deepspeed zero3_offload.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1 \
    --beta=${beta} \
    --gamma=1 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder "your_image_folder" \
    --video_folder "your_video_folder" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --unfreeze_mm_vision_tower False \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $TPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 1 \
    --learning_rate 3e-7 \
    --num_train_epochs 1 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True
   

