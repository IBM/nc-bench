import argparse
import glob
import os
import json
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, required=True)
    parser.add_argument('--generations_dir', type=str, required=True)
    parser.add_argument('--judgments_dir', type=str, required=True)
    parser.add_argument('--evaluations_dir', type=str, required=True)
    parser.add_argument('--reviews_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def assign_acceptable_answers(task):
    if task in ['inquiry', 'incremental request', 'self-correction', 'Self-Correction', 'Recommendation-Compact', 'Recommendation-Expanded']:
        return ['Answer', 'NonAnswer', 'Definition', 'RepeatRequest', 'ParaphraseRequest', 'ExampleRequest', 'DefinitionRequest']
    elif task in ['Recommendation-Incremental']:
        return ['Answer', 'NonAnswer', 'Definition', 'RepeatRequest', 'ParaphraseRequest', 'ExampleRequest', 'DefinitionRequest', 'DetailRequestGrounded']
    elif task in ['inquiry ungrounded', 'incremental-self-correction']:
        return ['NonAnswer', 'RepeatRequest', 'ParaphraseRequest', 'ExampleRequest', 'DefinitionRequest']
    elif task in ['repeat request', 'Repeat', 'Partial Repeat']:
        return ['Repeat', 'Partial Repeat']
    elif task in ['paraphrase request', 'Paraphrase']:
        return ['Paraphrase', 'Definition', 'Example']
    elif task in ['definition request', 'Definition']:
        return ['Definition']
    elif task in ['example request', 'Example']:
        return ['Example']
    elif ('sequence closer' in task) or ('Closer' in task):
        return ['PreClosing', 'Silence', 'NewTopic', 'NonVerbal', 'Acknowledgment', 'Acknowledgement', 'Assessment', 'GratitudeReceipt', 'AppreciationReceipt']
    elif task in ['sequence abort', 'Abort']:
        return ['PreClosing', 'Silence', 'GratitudeReceipt', 'Acknowledgment', 'NewTopic', 'Apology']
    elif 'Detail Request' in task:
        return ['DetailRequestGrounded']
    elif 'Preliminary' in task:
        return ['Affirmation', 'Acknowledgment', 'Assessment', 'HelpOffer', 'DetailRequestGrounded']
    elif task == 'Expansion-Choices':
        return ['ChoiceGiving']
    elif task == 'Expansion-Repair':
        return ['Repeat', 'Partial Repeat', 'Paraphrase', 'Example', 'Definition']
    else:
        return []


def evaluate_model(df):
    for index, row in df.iterrows():
        answers = row['output'] if isinstance(row['output'], str) else ''

        # split the output into individual answers
        answers_split = list(set([ans.strip() for ans in answers.replace('\n', ',').split(',') if ans]))
        
        # if 'Silence' or 'NonVerbal' is considered as answer, they must be the sole response.
        if 'Silence' in row['acceptable_answers'] or 'NonVerbal' in row['acceptable_answers']: 
            if 'Silence' in answers_split or 'NonVerbal' in answers_split:
                df.at[index, 'score'] = int(len(answers_split) == 1)
            else:
                df.at[index, 'score'] = int(any(ans in row['acceptable_answers'] for ans in answers_split))
        else:
            df.at[index, 'score'] = int(any(ans in row['acceptable_answers'] for ans in answers_split))

    return df


def aggregate_results(df):
    result = {}
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        result[task] = task_df['score'].sum() # raw score
        result[f'{task} count'] = len(task_df) # total number of instances for each task
        result[f'{task} ratio'] = result[task] / len(task_df) # correctness ratio by dividing the raw score by the total count
    return result


def main(args):
    files = glob.glob(os.path.join(args.judgments_dir, '*.jsonl'))
    generations = glob.glob(os.path.join(args.generations_dir, '*.jsonl'))
    prompt_df = pd.read_json(args.prompts_file, lines=True)
    
    for generation_file, input_file in zip(generations, files):
        filename = os.path.basename(input_file)
        filename_output = os.path.basename(generation_file)
        output_file = os.path.join(args.evaluations_dir, filename.replace('.jsonl', '.json'))

        df_generation = pd.read_json(generation_file, lines=True)
        df = pd.read_json(input_file, lines=True)
        df['acceptable_answers'] = df['task'].apply(assign_acceptable_answers)
        df = evaluate_model(df)
        print(len(df))
        generation_output = df_generation['output'].to_list()
        df.insert(2, 'model_response', generation_output)
        df.insert(2, 'freeform_prompt', prompt_df['freeform_prompt'].to_list())


        df.to_csv(args.reviews_dir + '/'+filename.replace('.jsonl', '.csv'), index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
