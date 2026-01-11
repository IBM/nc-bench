import argparse
import json
import os

import requests
import datasets
from tqdm import tqdm

from utils import generate_judge_prompts

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge', type=str, required=True, help="Name of the model to use for judge")
    parser.add_argument('--endpoint', type=str, required=True, help="API endpoint for model inference")
    parser.add_argument('--prompts_file', type=str, default='./data/conversation_competence.jsonl')
    parser.add_argument('--generations_file', type=str, required=True)
    parser.add_argument('--judgments_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    args = parser.parse_args()
    return args

def main(args):
    if not os.path.isfile(args.prompts_file):
        print(f'prompts file does not exist: {args.prompts_file}')
        return

    if not os.path.isfile(args.generations_file):
        print(f'generations file does not exist: {args.generations_file}')
        return

    if os.path.isfile(args.judgments_file):
        print(f'judgments file already exists: {args.judgments_file}')
        return

    dir_name = os.path.dirname(args.judgments_file)
    if not os.path.isdir(dir_name):
        print(f'judgments file directory does not exist: {dir_name}')
        return
    # load model
    # Prepare headers for API requests
    headers = {
        "accept": "application/json",
        "RITS_API_KEY": os.environ.get("RITS_API_KEY"),
        "Content-Type": "application/json"
    }

    chat_endpoint_path = "/v1/chat/completions"
    freeform_endpoint_path = "/v1/completions"


    # load judge prompts
    judge_prompts = generate_judge_prompts(
        prompts_file=args.prompts_file,
        generations_file=args.generations_file,
    )
    dataset = datasets.Dataset.from_list(judge_prompts)

    # get model judgments
    model_outputs = []
    for row in tqdm(dataset):
    	# Prepare payload for the API
        payload = {
            "messages": row['chat_prompt'],
            "model": args.judge,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
        }

        # Make the API call
        response = requests.post(url=f"{args.endpoint}{chat_endpoint_path}", json=payload, headers=headers)
        if response.status_code != 200:
            print(f"API call failed for ID {row.get('id', 'unknown')}: {response.status_code}, {response.text}")
            continue

        # Parse the API response
        try:
            response_data = response.json()
            output_content = response_data['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            print(f"Error parsing response for ID {row.get('id', 'unknown')}: {e}")
            continue

        # Store the model output
        model_output = {
            'id': row['id'],
            'task': row['task'],
            'output': output_content,
        }
        model_outputs.append(model_output)
        
    # export model judgments to file
    with open(args.judgments_file, 'w', encoding='utf8') as f:
        for row in model_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    args = get_args()
    main(args)
