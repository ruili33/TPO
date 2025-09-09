from PIL import Image
from io import BytesIO
import base64
import os
import math
import ast
import re
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
import numpy as np
import tempfile
import ffmpeg
from PIL import Image
import io
from transformers import StoppingCriteria
from typing import List, Optional, Union, Any
from llava.constants import IMAGE_TOKEN_INDEX

# Constants
SUPPORTED_PATCH_SIZES = [224, 336, 384, 448, 512]
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
DEFAULT_PLACEHOLDER_SIZE = (720, 720)
JPEG_START_MARKER = b'\xff\xd8'
JPEG_END_MARKER = b'\xff\xd9'
DEFAULT_RGB_COLOR = (0, 0, 0)
def resize_and_center_crop(image: Image.Image, shortest_edge_length: int) -> Image.Image:
    """Resize and center crop an image to a square with the specified edge length.
    
    Args:
        image: PIL Image to resize and crop
        shortest_edge_length: Target edge length for the square output
        
    Returns:
        Cropped PIL Image with dimensions (shortest_edge_length, shortest_edge_length)
    """
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image, grid_params):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image, data_args, processor=None):
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))
def opencv_extract_frames(
    vpath_or_bytesio: Union[str, BytesIO], 
    frames: int = 6, 
    fps: Optional[float] = None, 
    frame_count: Optional[int] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    start2: Optional[float] = None,
    end2: Optional[float] = None
) -> List[Image.Image]:
    """
    Extract frames from a video using OpenCV.

    Args:
        vpath_or_bytesio (str or BytesIO): Path to the video file or BytesIO object containing the video.
        frames (int): Number of frames to extract from the video.

    Returns:
        list: List of PIL Images extracted from the video.

    Raises:
        NotImplementedError: If the type of `vpath_or_bytesio` is not supported.
    """
    import cv2
    
    if isinstance(vpath_or_bytesio, str):
        if  os.path.isdir(vpath_or_bytesio):
            return read_images_from_directory(vpath_or_bytesio,frames,start)
        else:
            # vidcap = cv2.VideoCapture(vpath_or_bytesio)
            # return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
            
            return extract_frames(vpath_or_bytesio,frames,start,end,start2,end2)
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # assuming mp4
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
    else:
        raise NotImplementedError(type(vpath_or_bytesio))


def get_frame_from_vcap(
    vidcap: Any, 
    num_frames: int = 10, 
    fps: Optional[float] = None, 
    frame_count: Optional[int] = None
) -> List[Image.Image]:
    """Extract frames from video capture object.
    
    Args:
        vidcap: OpenCV video capture object
        num_frames: Number of frames to extract
        fps: Frames per second (computed if None)
        frame_count: Total frame count (computed if None)
        
    Returns:
        List of PIL Image frames
    """
    import cv2

    if fps == None or frame_count == None:
        # if one of fps or frame_count is None, still recompute
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Warning: Video file not found or invalid. Returning placeholder images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("Warning: Frame interval is 0 or video too short. Returning placeholder images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    # print("duration:", duration, "frames:", frame_count, "intervals:", frame_interval)

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count - 2, num_frames, dtype=int)
    retry=100
    while success or retry>0:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if not success:
            retry-=1
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                if len(images) >= num_frames:
                    return images
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            elif count >= 1:
                width, height = images[-1].size
                padding_needed = num_frames - len(images)
                images = [Image.new("RGB", (width, height))] * padding_needed + images
                print(f"Warning: Padding {padding_needed} frames for insufficient video length")
                return images
            else: 
                break
    raise ValueError("Failed to extract sufficient frames from video after retries")
def read_images_from_directory(directory: str, num_frames: int, index: Optional[int]) -> List[Image.Image]:
    """Read images from directory and sample to required number of frames.
    
    Args:
        directory: Path to directory containing images
        num_frames: Target number of frames to return
        index: Optional index parameter (currently unused)
        
    Returns:
        List of PIL Image objects
        
    Raises:
        ValueError: If no images found in directory
    """
    # Initialize lists for storing images and potential files
    images = []
    potential_file = []
    # Iterate through all files in directory
    for filename in sorted(os.listdir(directory)):
       
        file_path = os.path.join(directory, filename)
        
       
        if file_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            potential_file.append(file_path)
    # index=range(int(index/32*len(potential_file)),int((index+4)/32*len(potential_file)))
    # index=uniform_sample(index,sample_size=16)
    # if index is not None:
    #     assert max(index)<len(images)
    #     potential_file=[potential_file[j] for j in index]
    # print(len(potential_file))
    # print(potential_file)
    for file_path in potential_file:
            try:
                
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    images.append(img)
            except IOError:
                print(f"Warning: Failed to load or convert image at {file_path}")
    
    # Adjust number of frames to meet num_frames requirement
    total_images = len(images)
    if total_images == 0:
        raise ValueError("No images found in the directory.")
    
    if num_frames > total_images:
       
        width, height = images[-1].size
        images=[Image.new("RGB", (width, height))] * (num_frames - len(images))+images
        
    elif num_frames < total_images:
        # If fewer frames requested than available, sample uniformly
        indices = np.linspace(0, total_images - 1, num=num_frames, dtype=int)
        images = [images[i] for i in indices]
    
    # print(directory,num_frames,index)
    # print(images)
    return images

def find_jpeg_in_stream(stream: io.BytesIO):
    """Generator that finds and yields JPEG images from a byte stream.
    
    Args:
        stream: BytesIO stream containing JPEG data
        
    Yields:
        bytes: JPEG image data
    """
    start = b'\xff\xd8'
    end = b'\xff\xd9'
    while True:
        # Read data from current position to end of stream
        current_stream_content = stream.read()
        start_idx = current_stream_content.find(start)
        if start_idx == -1:
            break  # 没有找到更多的JPEG起始标记
        stream.seek(start_idx, io.SEEK_CUR)  # 调整流的位置到起始标记
        current_stream_content = current_stream_content[start_idx:]
        end_idx = current_stream_content.find(end)
        if end_idx == -1:
            break  # 没有找到结束标记，可能是流不完整
        # 获取整个JPEG图像
        jpeg_image = current_stream_content[:end_idx + 2]  # 包含结束标记
        stream.seek(end_idx + 2 - len(current_stream_content), io.SEEK_CUR)  # 调整流的位置到结束标记之后
        yield jpeg_image

def extract_frames(
    video_path: str, 
    num_frames: int, 
    start_time: Optional[float], 
    end_time: Optional[float], 
    start2: Optional[float], 
    end2: Optional[float]
) -> List[Image.Image]:
    """
Extract frames from two specific time ranges in a video while maintaining consistent frame rate.

    Args:
        video_path: Path to the video file
        num_frames: Total number of frames to extract
        start_time: Start time of first segment (relative time, range 0-1)
        end_time: End time of first segment (relative time, range 0-1)
        start2: Start time of second segment (relative time, range 0-1)
        end2: End time of second segment (relative time, range 0-1)

    Returns:
        List of PIL Image objects
    """
    # Get video information
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    
    # Calculate actual time ranges
    actual_start_time1 = duration * start_time
    actual_end_time1 = duration * end_time
    duration1 = actual_end_time1 - actual_start_time1
    if start2!=None and end2!=None:
        actual_start_time2 = duration * start2
        actual_end_time2 = duration * end2
        duration2 = actual_end_time2 - actual_start_time2

    # Calculate duration of both segments
    # Allocate frames proportionally by duration
        frames1 = int(num_frames * (duration1 / (duration1 + duration2)))
        frames2 = num_frames - frames1
    else:
        frames1=num_frames

   
    out1, _ = (
        ffmpeg
        .input(video_path, ss=actual_start_time1, t=duration1)
        .filter('fps', fps=frames1 / duration1)
        .output('pipe:', format='image2pipe', vframes=frames1, vcodec='mjpeg')
        .run(capture_stdout=True, capture_stderr=True)
    )
    byte_stream1 = io.BytesIO(out1)
    byte_stream1.seek(0)
    jpeg_images1 = list(find_jpeg_in_stream(byte_stream1))
    if start2!=None and end2!=None:
    
        out2, _ = (
            ffmpeg
            .input(video_path, ss=actual_start_time2, t=duration2)
            .filter('fps', fps=frames2 / duration2)
            .output('pipe:', format='image2pipe', vframes=frames2, vcodec='mjpeg')
            .run(capture_stdout=True, capture_stderr=True)
        )
        byte_stream2 = io.BytesIO(out2)
        byte_stream2.seek(0)
        jpeg_images2 = list(find_jpeg_in_stream(byte_stream2))

    # 将输出的字节数据转换为 PIL Image 对象列表
    

    

    # 将每个 JPEG 图像数据转换为 PIL Image 并合并两段的图像
        pil_images = [Image.open(io.BytesIO(jpeg_data)) for jpeg_data in (jpeg_images1 + jpeg_images2)]
    else:
        pil_images = [Image.open(io.BytesIO(jpeg_data)) for jpeg_data in jpeg_images1]

    return pil_images
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
