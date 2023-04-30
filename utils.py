import openai
import re
import json
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, load_from_disk
import time
import random
from openai.error import APIError, APIConnectionError, RateLimitError, Timeout
from datetime import datetime

random.seed(42)


def get_exemplar(dataset_name, prompt, shot, promptset):

    if promptset == '':
        promptset = dataset_name

    if prompt == 'zero-cot':
        return ''

    with open(f'exemplar_texts/{prompt}-{promptset}-{shot}shot.txt') as f:
        exemplar = f.read()

    return exemplar


def get_dataset(dataset_name, dev):

    if dataset_name == 'gsm8k':

        # Load from file for future edits
        dataset = load_from_disk("data/gsm8k")

    elif dataset_name == 'multiarith':

        dataset = load_from_disk("data/multiarith")

    elif dataset_name == 'aqua_rat':

        dataset = load_dataset(dataset_name, 'raw')["test"]

    else:
        raise ValueError("dataset is not properly defined ...")

    if dev:
        dataset = dataset.select(range(2))

    return dataset


def generate_prompt(question, exemplar, prompt):

    # an overall instruction could be added here if needed
    # instr = "End your response with 'The answer is <answer>.'"

    if prompt == 'pycot':
        prompt_text = exemplar + "\n\nQ: " + question + \
            " Write a Python function that returns the answer.\nA:\n"
    elif prompt == 'arithcot':
        prompt_text = exemplar + "\n\nQ: " + question + \
            " Write multiple mathematical equations to calculate the answer step by step. \nA:\n"
    elif prompt == 'sympy':
        prompt_text = exemplar + "\n\nQ: " + question + \
            "write a mapping and a mathematical equation starting with ‘Eq1:’ and solve using sympy"
    elif prompt == 'cot':
        prompt_text = exemplar + "\n\nQ: " + question + \
            "\nA:"
    elif prompt == 'zero-cot':
        prompt_text = "\n\nQ: " + question + \
            "\nA: Let's think step by step. "
    elif prompt == 'varcot':
        prompt_text = exemplar + "\n\nQ: " + question + \
            "\nA:"

    return prompt_text


def build_record(sample, result, mapping, dataset_name):

    record = {}
    record['question'] = sample['question']

    # record['answer'] = re.sub(
    #    r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", re.sub(r'<<.*?>>', '', sample['answer']))
    # record['numeric_answer'] = re.search(
    #    r"#### (\-?[0-9\.\,]+)", sample['answer']).group(1)
    if dataset_name == 'aqua_rat':
        record['answer'] = sample['correct']
    else:
        record['answer'] = sample['answer']

    if result['model'] == 'text-davinci-003':
        record['response'] = result['choices'][0]['text']
        # record['tokens'] = result['choices'][0]['logprobs']['tokens']
        # record['logprobs'] = result['choices'][0]['logprobs']['token_logprobs']

    elif result['model'] == 'text-davinci-002':
        record['response'] = result['choices'][0]['text']
        # record['tokens'] = result['choices'][0]['logprobs']['tokens']
        # record['logprobs'] = result['choices'][0]['logprobs']['token_logprobs']

    elif result['model'].startswith('gpt-3.5-turbo'):
        record['response'] = result['choices'][0]['message']['content']

    record['mapping'] = mapping

    return record


def evaluate_openai(run_id, model_name, dataset_name, prompt, shot, dev, promptset):
    with open(f'logs/{run_id}.jsonl', 'w') as f:

        # filename = 'logs/' + str(datetime.now()).replace(':','-') + '.jsonl'
        # with open(filename, 'w') as f:

        # retrieve the exemplar text
        exemplar = get_exemplar(dataset_name, prompt, shot, promptset)

        # retrieve the dataset
        dataset = get_dataset(dataset_name, dev)

        for sample in tqdm(dataset):

            # generate question text
            if dataset == 'aqua_rat':
                mapping = ''
                sample["question"] += ', '.join(sample['options'])
            elif prompt == 'zero-cot' or prompt == 'cot' or prompt =='sympy':
                mapping = ''
            else:
                sample["question"], mapping = extract_mapping(
                    sample["question"])

            # generate prompt text
            prompt_text = generate_prompt(sample["question"], exemplar, prompt)
            # get response
            result = generate_response(prompt_text, model_name)

            record = build_record(sample, result, mapping, dataset_name)
            f.write(json.dumps(record) + '\n')


# Function to interact with the model and generate a response
def generate_response(prompt, model_name):
    if model_name == 'gpt3':
        while True:
            try:
                response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=1,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout) as e:
                print(e)
                time.sleep(1)
                continue
            break
    elif model_name == 'text-davinci-002':
        while True:
            try:
                response = openai.Completion.create(
                    engine='text-davinci-002',
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=1,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout) as e:
                print(e)
                time.sleep(1)
                continue
            break
    elif model_name == 'gptturbo':
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout) as e:
                print(e)
                time.sleep(1)
                continue
            break

    return response

# Extract C0, C1, C2 Mapping from question


def extract_mapping(q):

    # Preprocess question to combine multiple number to same word
    q = re.sub(r'(\d)\s+(\d)', r'\1,\2', q)
    q = re.sub(r'(\d),(\d)', r'\1\2', q)

    # Seperate number and words
    q = re.sub(r'(\d)([a-z])', r'\1 \2', q)
    q = re.sub(r'(\d)([A-Z])', r'\1 \2', q)
    q = re.sub(r'(\d)(\-)', r'\1 \2', q)
    q = re.sub(r'\s+', ' ', q)

    # Expression used to extract number

    exp = r'\d+(\,\d+)*(\.\d+)?'

    mapping = {}
    new_question = []
    letter = 'c'
    count = 0
    name = letter + str(count)

    words = q.split(' ')

    # Replace number word with numbers

    # Finding numbers in the questions

    for word in words:

        # only support pure numbers, $, %, single number word, but not others (e.g.  1/4, 5th)

        # check if contains letter or other symbols
        if (bool(re.search('[a-zA-Z]', word))):
            new_question.append(word)
            continue

        if (bool(re.search('[\`\~\!\@\#\^\&\*\_\=\/\:\;]', word))):
            new_question.append(word)
            continue

        # checking first word
        try:
            # check if a number
            num = re.search(exp, word).group(0)
        except:
            new_question.append(word)
            continue

        # change question
        name = letter + str(count)
        mapping[name] = float(num)
        new_word = word.replace(num, name)
        new_question.append(new_word)
        count += 1

    return " ".join(new_question), mapping
