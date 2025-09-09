import json
import random
from tqdm import tqdm
import os
import re

import threading
from multiprocessing import Pool
import call_api
from openai import OpenAI

# Configuration parameters
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ""))

def count_words(text):
    """Count words in text using split() function"""
    words = text.split()
    return len(words)

def extract_qa(text):
    """Extract question and answer from formatted text using regex"""
    question_match = re.search(r'Q: (.+?)\n', text)
    answer_match = re.search(r'A: (.+)', text)
    
    # Extract question and answer
    question = question_match.group(1) if question_match else None
    answer = answer_match.group(1) if answer_match else None
    
    return question, answer

# Data configuration
caption_name = "caption_new_frames_4_total"
cur_type = "causual"
output_name = caption_name.replace("caption", "qa") + f"_{cur_type}_new"

# Load data
input_data = json.load(open(f"/pasteur/u/ruili/CogVLM2/{caption_name}.json"))
if os.path.exists(f"/pasteur/u/ruili/data_curation/{output_name}.json"):
    output = json.load(open(f"/pasteur/u/ruili/data_curation/{output_name}.json"))
else:
    output = {}


def generate(video_id):
    """Generate question-answer pairs for a given video"""
    cur_out = {}
    
    for selected_index in input_data[video_id]['id_list']:
        key_name = video_id + "&" + "{:02}".format(selected_index)
        
        if key_name in output:
            continue

        cur_caption = input_data[video_id]
    
        # Read prompt template
        with open(f'prompt/{cur_type}_caption.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Build prompt with frame captions
        prompt = ""
        for j in range(selected_index, selected_index + 4):
            frame_key = f'frame_{j:02d}.png'
            prompt += f"Caption for frame_{j:02d}: {cur_caption[frame_key].replace('image', 'frame')}\n"
        prompt += content
        
        ready_prompt = [
            {'role': 'system', 'content': "You are a helpful video question answer pair generator."},
            {'role': 'user', 'content': prompt}
        ]

        # Try to generate valid QA pairs
        tries = 0
        while tries < 2:
            # Generate QA pair
            response = call_api.call_api_2(
                prompt=ready_prompt,
                client=client,
                max_tokens=2000,
                temperature=0.8,
                timeout=180
            )
            
            # Check if response is valid
            if response is None or ("None" in response.choices[0].message.content or "none" in response.choices[0].message.content):
                tries += 1
                continue
                
            potential_question, _ = extract_qa(response.choices[0].message.content)
            if potential_question is None:
                tries += 1
                continue
            
            # Filter out common sense questions
            filter_prompt = "Golden Caption: \n"
            for j in range(32):
                if j in list(range(selected_index, selected_index + 4)) or j % 2 == 0:
                    frame_key = f'frame_{j:02d}.png'
                    filter_prompt += f"Caption for frame_{j:02d}: {cur_caption[frame_key]}\n"
            filter_prompt += f"\nQuestion: {potential_question}\n"
            filter_prompt += "Is this question about the video could be answered by the commonsense knowledge without taking into consideration of the visual and video information? Please answer with Yes or No."

            ready_prompt_filter = [
                {'role': 'system', 'content': "You are a helpful question judger."},
                {'role': 'user', 'content': filter_prompt}
            ]
            
            response_filter = call_api.call_api_2(
                prompt=ready_prompt_filter,
                client=client,
                max_tokens=2000,
                temperature=0,
                timeout=180
            )
            
            # Check if question requires visual information
            if response_filter is None or ("No" in response_filter.choices[0].message.content and "Yes" not in response_filter.choices[0].message.content):
                break
            else:
                tries += 1
                continue
                
        # Skip if couldn't generate valid QA pair
        if tries >= 2:
            continue 

        # Store the generated QA pair
        cur_out[key_name] = {
            "out": response.choices[0].message.content,
            "index_range": list(range(selected_index, selected_index + 4)),
            "commonsense": response_filter.choices[0].message.content,
            "video_id": video_id
        }

        # Extract and store question and answer separately
        input_text = cur_out[key_name]['out']
        question, answer = extract_qa(input_text)
        cur_out[key_name]['question'] = question
        cur_out[key_name]['answer'] = answer
        
    return cur_out


def main():
    """Main function: Execute question curation pipeline"""
    # Processing configuration
    num_processes = 200
    key_list = list(input_data.keys())
    
    # Batch process videos
    for ll in tqdm(range(0, len(key_list), num_processes)):
        with Pool(processes=num_processes) as pool:
            cur_result = pool.map(generate, key_list[ll:ll+num_processes])
        
        # Merge results
        for j in cur_result:
            output.update(j)
            
        # Save intermediate results
        with open(f"{output_name}.json", "w") as file:
            json.dump(output, file, indent=4)


if __name__ == "__main__":
    main()