accelerate launch --num_processes 6 --main_process_port 12331 -m lmms_eval \
    --model longva \
    --model_args pretrained=ruili0/LongVA-TPO,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=128,model_name=llava_qwen \
    --tasks videomme,mlvu,longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videomme_mlvu_longvideobench_val_v \
    --output_path ./logs/ 

accelerate launch --num_processes 6 --main_process_port 12331 -m lmms_eval \
    --model longva \
    --model_args pretrained=ruili0/LongVA-TPO,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=96,model_name=llava_qwen \
    --tasks videomme_w_subtitle \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videomme_w_subtitle \
    --output_path ./logs/ 
    