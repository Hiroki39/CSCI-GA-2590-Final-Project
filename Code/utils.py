import openai
import re
import json
from tqdm import tqdm
from datasets import concatenate_datasets
import time
import random
from openai.error import APIError, APIConnectionError, RateLimitError, Timeout

random.seed(42)


def get_exemplar():

    with open('exemplars.txt') as f:
        exemplar = f.read()

    return exemplar


def generate_prompt(question, exemplar):

    # an overall instruction could be added here if needed
    # instr = "End your response with 'The answer is <answer>.'"

    # prompt_text = instr + "\n\n" + exemplar + \
    #     "\n\nQ: " + question + "\nA:"
    prompt_text = exemplar + "\n\nQ: " + question + "\nA:"

    return prompt_text


def build_record(sample, result):

    record = {}
    record['question'] = sample['question']

    record['answer'] = re.sub(
        r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", re.sub(r'<<.*?>>', '', sample['answer']))
    record['numeric_answer'] = re.search(
        r"#### (\-?[0-9\.\,]+)", sample['answer']).group(1)

    if result['model'] == 'text-davinci-003':
        record['response'] = result['choices'][0]['text']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['text'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None
        record['tokens'] = result['choices'][0]['logprobs']['tokens']
        record['logprobs'] = result['choices'][0]['logprobs']['token_logprobs']

    elif result['model'].startswith('gpt-3.5-turbo'):
        record['response'] = result['choices'][0]['message']['content']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['message']['content'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None

    return record


def evaluate_openai(run_id, model_name, dataset):
    with open(f'logs/{run_id}.jsonl', 'w') as f:

        # generate exemplar
        exemplar = get_exemplar()

        # merge train and test datasets and remove the exemplar from the train set
        modified_ds = concatenate_datasets([dataset["train"].select(
            range(1, len(dataset["train"]))), dataset["test"]])

        for sample in tqdm(modified_ds):

            # generate question text
            question = sample["question"]
            # generate prompt text
            prompt_text = generate_prompt(question, exemplar)
            # get response
            result = generate_response(prompt_text, model_name)

            record = build_record(sample, result)
            f.write(json.dumps(record) + '\n')


# Function to interact with the model and generate a response
def generate_response(prompt, model_name):
    if model_name == 'gpt3':
        while True:
            try:
                response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=1,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout):
                time.sleep(1)
                continue
            break
    elif model_name == 'gptturbo':
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout):
                time.sleep(1)
                continue
            break

    return response
