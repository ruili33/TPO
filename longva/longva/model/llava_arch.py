#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
LLaVA Model Architecture Implementation

This module contains the core architecture classes for the LLaVA (Large Language and Vision Assistant) model,
including meta classes for vision-language integration and multimodal processing.
"""

# Standard library imports
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_resampler.builder import build_vision_resampler
from longva.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from longva.mm_utils import get_anyres_image_grid_shape
from longva.utils import rank0_print

# Constants
DEFAULT_SPATIAL_POOL_STRIDE = 3
DEFAULT_KERNEL_SIZE = 2
DEFAULT_SPATIAL_POOL_MODE = "average"


class LlavaMetaModel:
    """
    Meta class for LLaVA models that provides vision-language integration capabilities.
    
    This class serves as a mixin that adds multimodal processing capabilities to language models.
    It handles the integration of vision towers, resamplers, and projectors for processing
    visual inputs alongside text.
    
    Attributes:
        vision_tower: The vision encoder component
        vision_resampler: Component for resampling vision features
        mm_projector: Multimodal projector for aligning vision and language features
        image_newline: Special parameter for handling image newlines in unpad mode
    """

    def __init__(self, config) -> None:
        """
        Initialize the LlavaMetaModel with the given configuration.
        
        Args:
            config: Model configuration containing vision and multimodal settings
        """
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_llm(self):
        """
        Get the language model component.
        
        Returns:
            The language model, taking the first element if it's a list
        """
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm


    def get_vision_tower(self):
        """
        Get the vision tower component.
        
        Returns:
            The vision tower, taking the first element if it's a list
        """
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp: Optional[List] = None) -> None:
        """
        Initialize vision modules including tower, resampler, and projector.
        
        Args:
            model_args: Model arguments containing vision configuration
            fsdp: Optional FSDP (Fully Sharded Data Parallel) configuration
        """
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """
    Unpads a PyTorch tensor of a padded and resized image.

    This function removes padding that was added during image preprocessing to maintain
    aspect ratio while resizing to a fixed size.

    Args:
        tensor: The image tensor in CxHxW format
        original_size: The original size of the image as (width, height)

    Returns:
        The unpadded image tensor
        
    Raises:
        ValueError: If tensor dimensions are invalid
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (CxHxW), got {tensor.ndim}D tensor")
        
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    """
    Abstract base class for LLaVA causal language models.
    
    This class provides the interface and implementation for multimodal causal language modeling,
    combining vision and text processing capabilities for generation tasks.
    
    This is an abstract class that must be inherited by concrete implementations.
    """

    @abstractmethod
    def get_model(self):
        """
        Get the underlying model instance.
        
        Returns:
            The model instance
        """
        pass

    def get_vision_tower(self):
        """
        Get the vision tower from the underlying model.
        
        Returns:
            The vision tower component
        """
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D pooling to image features.
        
        This method reshapes image features into spatial dimensions and applies
        pooling (average or max) to reduce spatial resolution.
        
        Args:
            image_feature: Input image features with shape (num_frames, num_tokens, num_dim)
            
        Returns:
            Pooled image features with reduced spatial resolution
            
        Raises:
            ValueError: If mm_spatial_pool_mode is not 'average' or 'max'
        """
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        # Apply spatial pooling based on configuration
        pool_stride = getattr(self.config, 'mm_spatial_pool_stride', DEFAULT_SPATIAL_POOL_STRIDE)
        pool_mode = getattr(self.config, 'mm_spatial_pool_mode', DEFAULT_SPATIAL_POOL_MODE)
        
        if pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, pool_stride)
        elif pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, pool_stride)
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {pool_mode}. Expected 'average' or 'max'.")
            
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through the vision pipeline.
        
        Args:
            images: Input image tensor
            
        Returns:
            Encoded image features
        """
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    def encode_multimodals(
        self, 
        videos_or_images: torch.Tensor, 
        video_idx_in_batch: List[int], 
        split_sizes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Encode multiple modalities (images and videos) through the vision pipeline.
        
        Args:
            videos_or_images: Input tensor containing both videos and images
            video_idx_in_batch: Indices indicating which items are videos
            split_sizes: List of sizes to split the input tensor
            
        Returns:
            List of encoded features for each modality
        """
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []

        for idx, feat in enumerate(per_videos_or_images_features):
            feat = self.get_model().mm_projector(feat)
            # Apply 2D pooling for video features
            if idx in video_idx_in_batch:
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
        return all_videos_or_images_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        modalities: List[str] = ["image"],
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare inputs and labels for multimodal training/inference.
        
        This method processes multimodal inputs (text, images, videos) and prepares them
        for the language model by embedding images and integrating them with text tokens.
        
        Args:
            input_ids: Text token IDs
            position_ids: Position IDs for tokens
            attention_mask: Attention mask for valid tokens
            past_key_values: Cached key-value pairs from previous steps
            labels: Target labels for training
            images: Image or video tensors, can be a list or single tensor
            modalities: List of modality types ('image' or 'video')
            image_sizes: Original sizes of images before preprocessing
            
        Returns:
            Tuple of (input_ids, position_ids, attention_mask, past_key_values, 
                     input_embeds, labels) where input_embeds contains integrated
                     text and image embeddings
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            # Handle list of images or 5D tensor (batch of videos)
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            elif images.ndim == 5:
                images = images.flatten(0, 1)

            # Identify video indices in the batch
            video_idx_in_batch = []
            for idx, modality in enumerate(modalities):
                if modality == "video":
                    video_idx_in_batch.append(idx)

            # Normalize image dimensions
            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
                    
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            
            # Encode all modalities
            image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            
            # Apply patch merging strategy
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            
            elif mm_patch_merge_type == "unires":
                # Unified resolution processing for high-resolution images
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_idx in video_idx_in_batch:
                        # Video processing: flatten temporal and spatial dimensions
                        image_feature = image_feature.flatten(0, 1)
                    elif image_feature.shape[0] > 1:
                        # Multi-patch image processing
                        base_image_feature = image_feature[0]  # Base image not used in unires
                        image_feature = image_feature[1:]  # Process only high-res patches
                        
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0], \
                            f"Patch dimension mismatch: {height * width} != {base_image_feature.shape[0]}"
                            
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            
                        # Get grid shape for anyres processing
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx], 
                            self.config.image_grid_pinpoints, 
                            vision_tower_image_size
                        )
                        
                        # Reshape and pool the features
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        # Assume 2*2 patches
                        # After this, [2,2, 24,24, 4096]
                        kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                        kernel_size = 2
                        image_feature = image_feature.view(num_patch_height * num_patch_width, height, width, -1) # [4, 24, 24, 4096]
                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                        image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                        image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                        image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                        image_feature = image_feature.flatten(0, 1) # [576, 4096]
                        # rank0_print(f"After pool : {image_feature.shape}")
                    else:
                        # Text-only data: placeholder image feature
                        image_feature = image_feature[0]
                    new_image_features.append(image_feature)

                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
            """
            raise ValueError(error_message)
            # image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        # print("before_input_ids",input_ids.size())
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # Process each sequence in the batch
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            # Split text embeddings by image token positions
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            # Ensure all embeddings are on the correct device and concatenate
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make sequences longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences to the same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        # Apply position skipping augmentation during training if enabled
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
            
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer) -> None:
        """
        Initialize the tokenizer with vision-specific tokens.
        
        This method adds special tokens for image processing and configures
        embedding layers accordingly.
        
        Args:
            model_args: Model arguments containing tokenizer configuration
            tokenizer: The tokenizer to be initialized
        """
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}, "
                        f"Current: {input_embeddings.shape}, "
                        f"Number of new tokens: {num_new_tokens}"
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
