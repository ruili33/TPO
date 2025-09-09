"""Dataset module for LongVA multimodal learning.

This module contains dataset classes and utilities for DPO training
of multimodal models with support for images and videos.
"""

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from longva import conversation as conversation_lib
from longva.conversation import conv_templates, SeparatorStyle
from longva.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX
)
from longva.mm_utils import tokenizer_image_token, opencv_extract_frames, process_image
from longva.train.args import DataArguments, TrainingArguments


# Constants for dataset configuration
VIDEO_EXTENSIONS = (
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg',
    '.m4v', '.3gp', '.webm', '.vob', '.ogv', '.ts', '.m2ts', '.mts',
    '.f4v', '.rmvb', '.rm', '.divx', '.xvid', '.asf'
)

def preprocess_qwen(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    """Preprocess conversations using Qwen conversation template.
    
    Args:
        sources: List of conversation sources
        tokenizer: Tokenizer for text processing
        has_image: Whether the conversation includes images
        no_system_prompt: Whether to exclude system prompt
        
    Returns:
        Dictionary containing input_ids and labels
    """
    assert conversation_lib.default_conversation.version == "qwen"
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.CHATML

    # Mask targets
    sep = conv.sep +"\n"+ conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "" or rou=="\n":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # include <|eot_id|> for all rounds
            round_len += 1
            instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len and cur_len != total_len - 1:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,  # only work for v1
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    return  preprocess_qwen(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """Preprocess multimodal conversations by adding image tokens.
    
    Args:
        sources: List of conversation sources
        data_args: Data configuration arguments
        
    Returns:
        Processed sources with image tokens
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with lazy loading of multimodal data.
    
    This dataset supports both image and video data with lazy loading for efficient
    memory usage. It handles conversation formatting and multimodal preprocessing.
    
    Originally implemented by the LLaVA team and modified by Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        try:
            with open(data_path, "r") as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path, "r") as fp:
                list_data_dict = [json.loads(q) for q in fp]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        """Get length of each sample including image tokens."""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self) -> List[int]:
        """Get modality-aware lengths (positive for multimodal, negative for text-only)."""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    

    
    @staticmethod
    def _load_video(video_path: str, num_video_frames: int, data_args: DataArguments, 
                   fps: Optional[float] = None, frame_count: Optional[int] = None, 
                   index: Optional[int] = None, start: Optional[float] = None, 
                   end: Optional[float] = None, start2: Optional[float] = None, 
                   end2: Optional[float] = None):
        """Load video frames with error handling.
        
        Args:
            video_path: Path to video file
            num_video_frames: Number of frames to extract
            data_args: Data configuration arguments
            fps: Target FPS for extraction
            frame_count: Total frame count
            index: Frame index for extraction
            start: Start time ratio
            end: End time ratio
            start2: Second start time ratio (for multi-segment extraction)
            end2: Second end time ratio (for multi-segment extraction)
            
        Returns:
            Tuple of (PIL images list, success flag)
        """
        from longva.mm_utils import opencv_extract_frames
        
        video_loading_succeed = True
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
            
        try:
            pil_imgs = opencv_extract_frames(
                video_path, num_video_frames, fps, frame_count, 
                index, start=start, end=end, start2=start2, end2=end2
            )
        except Exception as e:
            video_loading_succeed = False
            print(f"Error loading video from {video_path}: {e}")
            # Create dummy images as fallback
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed
            


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(img, self.data_args, self.image_folder) for img in image_file]
                )
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            num_video_frames = self.data_args.num_video_frames
            if "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)
            if 'fps' in sources[0]:
                fps = sources[0]['fps']
            else:
                fps = None
            if 'frame_count' in sources[0]:
                frame_count = sources[0]['frame_count']
            else:
                frame_count = None

            images, video_loading_succeed = self._load_video(video_path, num_video_frames, self.data_args, fps=fps, frame_count=frame_count)

            image_tensor = torch.stack(
                [process_image(image, self.data_args, None) for image in images]
            )

            if "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                if isinstance(sources[0]["conversations"][1]["value"], str):
                    answer = sources[0]["conversations"][1]["value"].rstrip()
                else:
                    answer = str(sources[0]["conversations"][1]["value"]).rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]

            if not video_loading_succeed:
                answer = "Empty video."

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * num_video_frames + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict['labels'][:] = IGNORE_INDEX
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        return data_dict

def list_files_in_directory(directory_path: str) -> Dict[str, str]:
    """List video files in a directory and return a mapping of basename to full path.
    
    Args:
        directory_path: Path to the directory to scan
        
    Returns:
        Dictionary mapping file basenames (without extension) to absolute paths
    """
    files_dict = {}
    
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_base_name = os.path.splitext(filename)[0]
            absolute_path = os.path.abspath(os.path.join(directory_path, filename))
            if absolute_path.lower().endswith(VIDEO_EXTENSIONS):
                files_dict[file_base_name] = absolute_path
    
    return files_dict


class LazySupervisedDPODataset(LazySupervisedDataset):
    """Dataset for Direct Preference Optimization (DPO) training.
    
    Extends LazySupervisedDataset to support DPO training with chosen/rejected pairs.
    Handles video data with temporal segment sampling and preference-based training.
    
    Originally implemented by the LLaVA team and modified by Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_processor,
        model_cfg
    ):
        super(LazySupervisedDataset, self).__init__()
        try:
            with open(data_path, "r") as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path, "r") as fp:
                list_data_dict = [json.loads(q) for q in fp]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        self.img_processor = image_processor
        self.model_cfg = model_cfg
        # Initialize reference log probabilities for DPO training
        prob = {i: {'reference_chosen_logps': None, "reference_rejected_logps": None} 
                for i in range(len(self.list_data_dict))}
        self.logp = prob
        random.seed(42)
        self.file_dict = {}
        
        # Build file dictionary from all image folders
        for folder in image_folder:
            self.file_dict.update(list_files_in_directory(folder))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    

    
    @staticmethod
    def _load_video(video_path, num_video_frames, data_args, fps=None, frame_count=None,start=None,end=None):
        from longva.mm_utils import opencv_extract_frames
        video_loading_succeed = True
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
        try:
            pil_imgs = opencv_extract_frames(video_path, num_video_frames, fps, frame_count,start=start,end=end)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed
            


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(img, self.data_args, self.image_folder) for img in image_file]
                )
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            assert "&" in sources[0]['id']
            video_frame_index = int(sources[0]['id'].split("&")[1])
            num_video_frames = self.data_args.num_video_frames
            video_path = self.file_dict[sources[0]["video"]]
            if 'fps' in sources[0]:
                fps = sources[0]['fps']
            else:
                fps = None
            if 'frame_count' in sources[0]:
                frame_count = sources[0]['frame_count']
            else:
                frame_count = None
            # Load video frames from start to end
            images, video_loading_succeed = self._load_video(
                video_path, num_video_frames, self.data_args, 
                fps=fps, frame_count=frame_count, start=0, end=1
            )
            image_tensor = torch.stack([
                process_image(image, self.img_processor, self.model_cfg, is_video=True) 
                for image in images
            ])
            
            question = sources[0]["prompt"]
            chosen = sources[0]["chosen"]
            rejected = sources[0]["rejected"]

            if not video_loading_succeed:
                answer = "Empty video."

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * len(images) + question
            conversation_chosen = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": chosen},
            ]
            conversation_reject = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": rejected},
            ]

            sources_chosen = [conversation_chosen]
            sources_reject = [conversation_reject]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict_chosen = preprocess(
            sources_chosen,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        data_dict_reject = preprocess(
            sources_reject,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        chosen_data_dict = {k: v[0] for k, v in data_dict_chosen.items()}
        rejected_data_dict = {k: v[0] for k, v in data_dict_reject.items()}
        data_dict = dict()
        if isinstance(i, int):
            for k, toks in {
                "chosen": chosen_data_dict,
                "rejected": rejected_data_dict,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    data_dict[f"{k}_{type_key}"] = tokens
        
        
            
        # print("dataset",image_tensor.size(),flush=True)
        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
            data_dict['modalities']=['image']
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict['labels'][:] = IGNORE_INDEX
            data_dict['modalities']=['video']*num_video_frames
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        if self.logp[i]==None:
            reference_chosen_logp,reference_rejected_logp=None,None
        else:
            data_dict['reference_chosen_logp'],data_dict['reference_rejected_logp']=self.logp[i]['reference_chosen_logps'],self.logp[i]['reference_rejected_logps']
        # print(data_dict["image"].size())
        
        return data_dict



@dataclass
class DataCollatorForSupervisedDatasetDPO(object):
    
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        chosen_input_ids,rejected_input_ids, chosen_labels,rejected_labels, images,modalities,reference_chosen_logp,reference_rejected_logp = [],[],[], [], [],[],[],[]
        for instance in instances:
            if not isinstance(instance["chosen_input_ids"], list):
                chosen_input_ids.append(instance["chosen_input_ids"])
            else:
                chosen_input_ids += instance["chosen_input_ids"]
            if not isinstance(instance["chosen_labels"], list):
                chosen_labels.append(instance["chosen_labels"])
            else:
                chosen_labels += instance["chosen_labels"]
            if not isinstance(instance["rejected_input_ids"], list):
                rejected_input_ids.append(instance["rejected_input_ids"])
            else:
                rejected_input_ids += instance["rejected_input_ids"]
            if not isinstance(instance["rejected_labels"], list):
                rejected_labels.append(instance["rejected_labels"])
            else:
                rejected_labels += instance["rejected_labels"]
            if not isinstance(instance["modalities"], list):
                modalities.append(instance["modalities"])
            else:
                modalities += instance["modalities"]
            reference_chosen_logp.append(instance['reference_chosen_logp'])
            reference_rejected_logp.append(instance['reference_rejected_logp'])
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                # print(cur_image.shape)
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["chosen_input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _input_ids in zip(images, chosen_input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"
        for _images, _input_ids in zip(images, rejected_input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=IGNORE_INDEX)
        rejected_labels = torch.nn.utils.rnn.pad_sequence(rejected_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        if None in reference_chosen_logp:
            reference_chosen_logp_out=None
            reference_rejected_logp_out=None
        else:
            reference_chosen_logp_out=torch.tensor(reference_chosen_logp)
            reference_rejected_logp_out=torch.tensor(reference_rejected_logp)

        chosen_input_ids = chosen_input_ids[:, : self.tokenizer.model_max_length]
        rejected_input_ids = rejected_input_ids[:, : self.tokenizer.model_max_length]

        chosen_labels = chosen_labels[:, : self.tokenizer.model_max_length]
        rejected_labels = rejected_labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            rejected_input_ids=rejected_input_ids,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
            reference_chosen_logps=reference_chosen_logp_out,
            reference_rejected_logps=reference_rejected_logp_out
        )
        # print(batch)
        new_images = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(chosen_input_ids)):
            num_images = (chosen_input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            # print(num_images)
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images)
        
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
        batch['modalities']=modalities 
        return batch




class LazySupervisedEvaluateDataset(LazySupervisedDataset):
    """Dataset for evaluation with multimodal data.
    
    Specialized dataset for evaluation tasks with support for flexible video
    frame sampling and prompt-based inference.
    
    Originally implemented by the LLaVA team and modified by Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        list_data_dict,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_processor,
        model_cfg,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict=list_data_dict
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.img_processor=image_processor
        # prob={i:{'reference_chosen_logps':1,"reference_rejected_logps":1} for i in range(len(self.list_data_dict))}
        # self.logp=prob
        self.model_cfg=model_cfg
        random.seed(42)
        self.file_dict={}
        # print(image_folder,"image_folder",flush=True)
        # for i in image_folder:
        #     self.file_dict.update(list_files_in_directory(i))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    

    
    @staticmethod
    def _load_video(video_path, num_video_frames, data_args, fps=None, frame_count=None,start=None,end=None,start2=None,end2=None):
        from longva.mm_utils import opencv_extract_frames
        video_loading_succeed = True
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
        try:
            pil_imgs = opencv_extract_frames(video_path, num_video_frames, fps, frame_count,start=start,end=end,start2=start2,end2=end2)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed
            


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(img, self.data_args, self.image_folder) for img in image_file]
                )
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            # assert "&" in sources[0]['id']
            # video_frame_index=int(sources[0]['id'].split("&")[1])
            # num_video_frames = self.data_args.num_video_frames
            # if "video" in sources[0]:
            #     video_file = sources[0]["video"]
            # else:
            #     video_file = sources[0]["video_id"] + ".mp4"
            # video_folder = self.image_folder
            video_path =sources[0]["video"]
            num_video_frames=sources[0]["num_frames"]
            # video_path = os.path.join(video_folder, video_file)
            if 'fps' in sources[0]:
                fps = sources[0]['fps']
            else:
                fps = None
            if 'frame_count' in sources[0]:
                frame_count = sources[0]['frame_count']
            else:
                frame_count = None
            start=sources[0]['start']
            end=sources[0]['end']
            start2=sources[0]['start2'] if "start2" in sources[0] else None
            end2=sources[0]['end2'] if "end2" in sources[0] else None
            images, video_loading_succeed = self._load_video(
                video_path, num_video_frames, self.data_args, 
                fps=fps, frame_count=frame_count, start=start, end=end, 
                start2=start2, end2=end2
            )
            # print("image",images[0].size)
            if 'frame_index' in sources[0]:
                images=[images[i] for i in sources[0]['frame_index'] if i<len(images)]
            # print(len(images))
            image_tensor = torch.stack(
                [process_image(image, self.img_processor, self.model_cfg,is_video=True) for image in images]
            )
            # print("image_tensor",image_tensor.size())
            question = sources[0]["prompt"]
            # chosen = sources[0]["chosen"]
            # rejected = sources[0]["rejected"]

            # video_loading_succeed=False
            if not video_loading_succeed:
                answer = "Empty video."

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * len(images) + question
            # conversation_chosen = [
            #     {"from": "human", "value": question},
            #     {"from": "gpt", "value": None},
            # ]
            conv = conv_templates[self.data_args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            input_ids = torch.unsqueeze(input_ids, 0)
            input_ids = torch.as_tensor(input_ids)
            # conversation_reject = [
            #     {"from": "human", "value": question},
            #     {"from": "gpt", "value": rejected},
            # ]

            # sources_chosen = [conversation_chosen]
            # sources_reject = [conversation_reject]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        # data_dict_chosen = preprocess(
        #     sources_chosen,
        #     self.tokenizer,
        #     has_image=(
        #         "image" in self.list_data_dict[i]
        #         or "video" in self.list_data_dict[i]
        #         or "video_id" in self.list_data_dict[i]
        #     ),
        # )
        data_dict_chosen=dict(input_ids=input_ids)
        # data_dict_reject = preprocess(
        #     sources_reject,
        #     self.tokenizer,
        #     has_image=(
        #         "image" in self.list_data_dict[i]
        #         or "video" in self.list_data_dict[i]
        #         or "video_id" in self.list_data_dict[i]
        #     ),
        # )
        chosen_data_dict = {k: v[0] for k, v in data_dict_chosen.items()}
        # rejected_data_dict = {k: v[0] for k, v in data_dict_reject.items()}
        data_dict = chosen_data_dict
        # if isinstance(i, int):
        #     for k, toks in {
        #         "chosen": chosen_data_dict,
        #     }.items():
        #         for type_key, tokens in toks.items():
        #             if type_key == "token_type_ids":
        #                 continue
        #             data_dict[f"{type_key}"] = tokens
        
        
            
        # print("dataset",image_tensor.size(),flush=True)
        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
            data_dict['modalities']=['image']
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict['labels'][:] = IGNORE_INDEX
            data_dict['modalities']=['video']*num_video_frames
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        # if self.logp[i]==None:
        #     reference_chosen_logp,reference_rejected_logp=None,None
        # else:
        #     data_dict['reference_chosen_logp'],data_dict['reference_rejected_logp']=self.logp[i]['reference_chosen_logps'],self.logp[i]['reference_rejected_logps']
        # print(data_dict["image"].size())
        data_dict['index']=sources[0]["id"]
        return data_dict


@dataclass
class DataCollatorForSupervisedDatasetEvaluate(object):
    
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        input_ids, images,modalities,index = [],[], [],[]
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            
            if not isinstance(instance["modalities"], list):
                modalities.append(instance["modalities"])
            else:
                modalities += instance["modalities"]
            index.append(instance["index"])
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                # print(cur_image.shape)
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        

        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            index=index
        )

        new_images = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            # print(num_images)
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images.unsqueeze(0))
        
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
        batch['modalities']=modalities 
        
        # print("batch[images]",batch["images"].size())
        return batch
def get_image_folders(image_folder) -> Optional[List[str]]:
    if image_folder is None:
        return None
    elif isinstance(image_folder, str):
        if ',' in image_folder:
            return [folder.strip() for folder in image_folder.split(',') if folder.strip()]
        else:
            return [image_folder]
    elif isinstance(image_folder, list):
        return image_folder
    else:
        raise ValueError(f"image_folder must be str or List[str], got {type(image_folder)}")

def make_dpo_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
    model_cfg,
    image_processor,
    data_path: Optional[str] = None,
    image_folders: Optional[List[str]] = None
) -> Dict:
    """Make dataset and collator for DPO training.
    
    Args:
        tokenizer: The tokenizer to use for preprocessing
        data_args: Data configuration arguments
        training_args: Training configuration arguments
        model_cfg: Model configuration
        image_processor: Image processor for preprocessing
        data_path: Path to the training data file (JSONL format)
        image_folders: List of folders containing images/videos
    
    Returns:
        Dictionary containing train_dataset, eval_dataset, and data_collator
    """
    # Use provided paths or fall back to data_args
    if data_path is None:
        data_path = getattr(data_args, 'data_path', None)
    if image_folders is None:
        image_folders = get_image_folders(data_args.image_folder)
    
    if data_path is None:
        raise ValueError("data_path must be provided either as argument or in data_args")
    if image_folders is None:
        raise ValueError("image_folders must be provided either as argument or in data_args")
    
    train_dataset = LazySupervisedDPODataset(
        data_path=data_path,
        image_folder=image_folders,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_cfg=model_cfg,
        image_processor=image_processor
    )
    
    print(f"Dataset length: {len(train_dataset)}")
    
    data_collator = DataCollatorForSupervisedDatasetDPO(
            tokenizer,
            data_args=data_args
        )
    
    training_args.sample_lens = [len(train_dataset)]
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
