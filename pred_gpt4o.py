import argparse
import json
import os

import openai
import datasets
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06')
    parser.add_argument('--prompts_file', type=str, default='./data/conversation_competence.jsonl')
    parser.add_argument('--generations_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    args = parser.parse_args()
    return args


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

    # create client
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # load dataset
    dataset = datasets.load_dataset('json', data_files=args.prompts_file, split='train')

    # get model generations
    model_outputs = []
    for row in tqdm(dataset):
        # generate
        completion = client.chat.completions.create(
            model=args.model,
            messages=row['chat_prompt'],
            max_completion_tokens=args.max_new_tokens,
            temperature=0.0,
        )

        # store outputs
        model_output = {
            'id': row['id'],
            'task': row['task'],
            'output': completion.choices[0].message.content,
        }
        model_outputs.append(model_output)

    # export model generations to file
    with open(args.generations_file, 'w', encoding='utf8') as f:
        for row in model_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
