import argparse
import json
import os
import requests
from tqdm import tqdm
import datasets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Name of the model to use for inference")
    parser.add_argument('--endpoint', type=str, required=True, help="API endpoint for model inference")
    parser.add_argument('--prompts_file', type=str, default='./data/conversation_competence.jsonl', help='File path to load input prompts')
    parser.add_argument('--generations_file', type=str, required=True, help="File path to save the model generations")
    parser.add_argument('--max_tokens', type=int, default=128, help="Maximum number of tokens to generate")
    return parser.parse_args()


def main(args):
    if not os.path.isfile(args.prompts_file):
        print(f'prompts file does not exist: {args.prompts_file}')
        return

    if os.path.isfile(args.generations_file):
        print(f'generations file already exists: {args.generations_file}')
        return

    dir_name = os.path.dirname(args.generations_file)
    if not os.path.isdir(dir_name):
        print(f'generations file directory does not exist: {dir_name}')
        return

    # Load dataset
    dataset = datasets.load_dataset('json', data_files=args.prompts_file, split='train')

    # Prepare headers for API requests
    headers = {
        "accept": "application/json",
        "RITS_API_KEY": os.environ.get("RITS_API_KEY"),
        "Content-Type": "application/json"
    }

    chat_endpoint_path = "/v1/chat/completions"
    freeform_endpoint_path = "/v1/completions"

    # Get model generations via API requests
    model_outputs = []
    for row in tqdm(dataset):
        # Prepare payload for the API
        payload = {
            "messages": row['chat_prompt'],
            "model": args.model,
            "max_tokens": args.max_tokens,
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

    # Export model generations to file
    with open(args.generations_file, 'w', encoding='utf8') as f:
        for row in model_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
