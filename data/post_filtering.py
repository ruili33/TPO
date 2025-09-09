import json
import os
from tqdm import tqdm
from openai import OpenAI
from multiprocessing import Pool
import call_api

# Configuration parameters
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ""))

# Data path configuration

longva_out = "your_longva_output"
qa = "your_question_curation_outputs"
caption_name = "your_caption_outputs"

# Load data
input_data_preferred = json.load(open(longva_out))
input_data_dispreferred = json.load(open(longva_out))
golden = json.load(open(qa))
caption = json.load(open(caption_name))

qa_out = qa + f"_{folder.lower()}"

# Initialize result dataset
if os.path.exists(f"output/{qa_out}_preference_combined.json"):
    preference_dataset = {}
else:
    preference_dataset = {}

def filter(ind):
    """
    Filter function: Compare quality of two answers and determine if they meet requirements
    
    Args:
        ind: Data index
    
    Returns:
        Dictionary of qualified data, empty dict if not qualified
    """
    cur_preference_dataset = {}
    
    # Skip already processed data or non-existing data
    if ind in preference_dataset or ind not in input_data_dispreferred:
        return {}
    
    # Get data
    preferred = input_data_preferred[ind]
    dispre = input_data_dispreferred[ind]
    question = golden[ind]['question']
    video_id = golden[ind]['video_id']
    cur_caption = caption[video_id]
    
    # Build prompt
    prompt = "Golden Caption: \n"
    for j in range(32):
        if j % 2 == 0:
            frame_key = f'frame_{j:02d}.png'
            prompt += f"Caption for frame_{j:02d}: {cur_caption[frame_key]}\n"
    
    prompt += f"Question: {question}\n\n"
    prompt += f"Answer1: {preferred}\n\n"
    prompt += f"Answer2: {dispre}\n\n"
    
    prompt += "You are given a question about a given video and its captions, along with two potential answers.\n"
    prompt += "Task1: Is Answer1 better than Answer2? Please answer with Yes or No or Equally.\n"
    prompt += "Task2: Please check if this question and Answer1 contradicts to any part of the golden caption or this question might have another answer different from Answer1. Please respond with Yes or No.\n"
    prompt += "Task3: Is the Answer1 is correct given the question and golden caption? Please answer with Yes or No.\n"
    prompt += "Please answer in the format: Task1: [Yes or No or Equally]\nTask2: [Yes or No]\nTask3: [Yes or No]."
    
    # Call API
    ready_prompt = [
        {'role': 'system', 'content': "You are a helpful answer judger."},
        {'role': 'user', 'content': prompt}
    ]
    
    response = call_api.call_api_2(
        prompt=ready_prompt,
        client=client,
        max_tokens=2000,
        temperature=0,
        timeout=180,
        model="gpt-4o-2024-08-06"
    )
    
    # Parse response
    if response and response.choices and response.choices[0].message:
        response_text = response.choices[0].message.content.lower()
        response_list = response_text.split("task")
        
        # Check if all conditions are met: Answer1 is better, no contradiction, and correct
        if (len(response_list) >= 4 and
            "yes" in response_list[1] and "no" not in response_list[1] and "equal" not in response_list[1] and
            "no" in response_list[2] and "yes" not in response_list[2] and
            "yes" in response_list[3] and "no" not in response_list[3]):
            
            golden[ind]['preferred'] = preferred
            golden[ind]['reject'] = dispre
            cur_preference_dataset[ind] = golden[ind]
    
    return cur_preference_dataset


def main():
    """Main function: Execute data filtering pipeline"""
    # Processing configuration
    input_data_preferred_key_list = list(input_data_preferred.keys())
    num_processes = 200
    
    # Batch process data
    print(f"Starting to process {len(input_data_preferred_key_list)} data entries...")
    for ll in tqdm(range(0, len(input_data_preferred_key_list), num_processes)):
        with Pool(processes=num_processes) as pool:
            cur_result = pool.map(filter, input_data_preferred_key_list[ll:ll+num_processes])
        
        # Merge results
        for j in cur_result:
            preference_dataset.update(j)
        
        # Save intermediate results
        output_file = f"output/{qa_out}_preference_combined.json"
        os.makedirs("output", exist_ok=True)
        with open(output_file, "w") as file:
            json.dump(preference_dataset, file, indent=4)
    
    print(f"Processing completed. Retained {len(preference_dataset)} qualified data entries.")


if __name__ == "__main__":
    main()