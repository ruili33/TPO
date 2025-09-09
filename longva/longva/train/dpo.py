import sys
from longva.model.builder import load_pretrained_model
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import logging
from decord import VideoReader, cpu
import torch
import numpy as np
import warnings
import os
from transformers import HfArgumentParser,AutoConfig
from longva.train.args import TrainingArguments, ModelArguments, DataArguments
from transformers import set_seed
from longva.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    vision_resolution_elevation,
    unit_test_rope_scaling,
    mprint,
)
from longva.model.language_model.llava_qwen import LlavaQwenConfig
from longva.data import make_dpo_data_module

from longva.train.llava_trainer import LLaVADPOTrainer
from longva.train.callbacks.autoresume_callback import AutoResumeCallback
import longva.conversation as conversation_lib
from enum import Enum
class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"

os.environ["WANDB__SERVICE_WAIT"] = "300"
warnings.filterwarnings("ignore", message="Caution: Your LLM is currently in training mode")
local_rank = None
if "WANDB_PROJECT" not in os.environ:
    # Default to WANDB project "VILA".
    os.environ["WANDB_PROJECT"] = "longva_dpo"

def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    print("local_rank",local_rank)
    torch.cuda.set_device(local_rank)
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    bnb_model_from_pretrained_args = {}
    set_seed(training_args.seed)
    # set_seed(200)
    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)


    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
    else:
        resume_from_checkpoint = False
        config = LlavaQwenConfig.from_pretrained(model_args.model_name_or_path)
    
    model_path = "lmms-lab/LongVA-7B"
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen")
    config = model.config
    # print(config)
    # model.
    if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    prepare_config_for_training(config, model_args, training_args, data_args)

    # vision_resolution_elevation(model, config)


    # Take a look on model architecture.
    mprint(model)

    # model.llm.config.use_cache = False
    ## set tunnable parameters
    logging.warning(
        "You are setting tunable parameters for the model. Previous args include 'freeze_backbone' and 'tune_mm_mlp_adapter' are deprecated.\n Notice: default value of tune_xxx is False, which means you would not tune this part."
    )

    model.model.embed_tokens.requires_grad_(training_args.tune_language_model)
    model.model.layers.requires_grad_(training_args.tune_language_model)


    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
    if model.model.vision_tower:
        model.model.vision_tower.requires_grad_(training_args.tune_vision_tower)
        model.model.mm_projector.requires_grad_(training_args.tune_mm_projector)
        mprint(f"vision tower {training_args.tune_vision_tower}")
        mprint(f"mm projector {training_args.tune_mm_projector}")

    if not any([training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False
    conversation_lib.default_conversation = conversation_lib.conv_templates[
                "qwen_1_5"
            ]
    # if need_to_modify_do_sample(model.llm.generation_config):
    #     model.model.llm.generation_config.do_sample = True

    # if training_args.gradient_checkpointing:
    #     if hasattr(model.llm, "enable_input_require_grads"):
    #         model.llm.enable_input_require_grads()
    #     else:

    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)

    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # print(training_args)
    data_args.image_processor=image_processor
    data_args.is_multimodal = True
    data_module = make_dpo_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_cfg=config,
        image_processor=image_processor
    )

    training_args.model_init_kwargs=None
    training_args.ref_model_init_kwargs=None
    training_args.model_adapter_name=None
    training_args.ref_adapter_name=None
    training_args.reference_free=False
    training_args.precompute_ref_log_probs=False
    training_args.max_length=None
    training_args.max_prompt_length=None
    training_args.max_target_length=None
    training_args.label_pad_token_id= -100
    training_args.disable_dropout=True
    training_args.generate_during_eval=False
    training_args.padding_value=None
    training_args.truncation_mode="keep_end"
    training_args.loss_type="sigmoid"
    training_args.label_smoothing= 0
    training_args.f_divergence_type= FDivergenceType.REVERSE_KL
    training_args.f_alpha_divergence_coef= 1.0
    training_args.dataset_num_proc= None
    training_args.sync_ref_model= False
    # training_args.rpo_alpha=  None

    callbacks = [AutoResumeCallback()]
    # ref_model=deepcopy(model)
    # _, ref_model, _, _ = load_pretrained_model(model_path, None, "llava_qwen")
    # ref_model.eval()
    # print(training_args)
    if training_args.lora_enable==True:
        from peft import LoraConfig
        from peft import get_peft_model
        peft_config = LoraConfig(
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                r=training_args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules= "model.(layers.*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj)|mm_projector.*(0|2)).*$"
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        peft_config=None
    print("model,before",model,flush=True)
    trainer = LLaVADPOTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        callbacks=callbacks, **data_module
    )

    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )
    
    

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.save_model()
    trainer.save_state()

    model.config.resume_path = model.config._name_or_path = training_args.output_dir

    safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
    





# # fix seed
# torch.manual_seed(0)


# max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
# gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
# # you can also set the device map to auto to accomodate more frames



# #image input
# prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDescribe the image in details.<|im_end|>\n<|im_start|>assistant\n"
# input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
# image = Image.open(image_path).convert("RGB")
# images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
# with torch.inference_mode():
#     output_ids = model.generate(input_ids, images=images_tensor, image_sizes=[image.size], modalities=["image"], **gen_kwargs)
# outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
# print(outputs)
# print("-"*50)

# #video input
# prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nGive a detailed caption of the video as if I am blind.<|im_end|>\n<|im_start|>assistant\n"
# input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
# vr = VideoReader(video_path, ctx=cpu(0))
# total_frame_num = len(vr)
# uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
# frame_idx = uniform_sampled_frames.tolist()
# frames = vr.get_batch(frame_idx).asnumpy()
# video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
# with torch.inference_mode():
#     output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
# outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
# print(outputs)