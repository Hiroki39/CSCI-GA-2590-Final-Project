import openai
import re
import json
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset
import time
import random
from openai.error import APIError, APIConnectionError, RateLimitError, Timeout

# For mapping extraction
from word2number import w2n

random.seed(42)


def get_exemplar(dataset_name, prompt, shot):

    with open(f'exemplar_texts/{prompt}-{dataset_name}-{shot}shot.txt') as f:
        exemplar = f.read()

    return exemplar


def get_dataset(dataset_name):

    # Load the GSM8K dataset from Hugging Face
    if dataset_name == 'gsm8k':
        # Load the GSM8K dataset from Hugging Face
        dataset = load_dataset(dataset_name, 'main')
    else:
        pass

    return dataset


def generate_prompt(question, exemplar, prompt):

    # an overall instruction could be added here if needed
    # instr = "End your response with 'The answer is <answer>.'"

    if prompt == 'pycot':
        prompt_text = exemplar + "\n\nQ: " + question + \
            " Write a Python function that returns the answer.\nA:"
    elif prompt == 'sympy':
        pass

    return prompt_text


def build_record(sample, result, mapping):

    record = {}
    record['question'] = sample['question']

    record['answer'] = re.sub(
        r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", re.sub(r'<<.*?>>', '', sample['answer']))
    record['numeric_answer'] = re.search(
        r"#### (\-?[0-9\.\,]+)", sample['answer']).group(1)

    if result['model'] == 'text-davinci-003':
        record['response'] = result['choices'][0]['text']
        record['tokens'] = result['choices'][0]['logprobs']['tokens']
        record['logprobs'] = result['choices'][0]['logprobs']['token_logprobs']

    elif result['model'].startswith('gpt-3.5-turbo'):
        record['response'] = result['choices'][0]['message']['content']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['message']['content'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None
    
    record['mapping'] = mapping

    return record


def evaluate_openai(run_id, model_name, dataset_name, prompt, shot, dev):
    with open(f'logs/{run_id}.jsonl', 'w') as f:

        # retrieve the exemplar text
        exemplar = get_exemplar(dataset_name, prompt, shot)
        # retrieve the dataset
        dataset = get_dataset(dataset_name)

        if not dev:
            # merge train and test datasets and remove the exemplar from the train set
            modified_ds = concatenate_datasets([dataset["train"].select(
                range(shot, len(dataset["train"]))), dataset["test"]])
        else:
            modified_ds = dataset["test"].select(range(5))

        for sample in tqdm(modified_ds):

            # generate question text
            new_question, mapping = extract_mapping(sample["question"])
            # generate prompt text
            prompt_text = generate_prompt(new_question, exemplar, prompt)
            # get response
            result = generate_response(prompt_text, model_name)

            record = build_record(sample, result, mapping)
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

# Extract ABC Mapping from question
def extract_mapping(q):
    
    # Preprocess question
    
    q = re.sub(r'(\d)\s+(\d)', r'\1,\2', q)
    q = re.sub(r'(\d),(\d)', r'\1\2', q)
    
    # Expression used to extract number

    exp = r'\d+(\,\d+)*(\.\d+)?'
    
    mapping = {}
    new_question = q
    letter = 'A'

    words = q.split(' ')
    
    # Replace number word with numbers

    # Finding numbers in the questions

    for j, word in enumerate(words):
            
        # only support pure numbers, price, %, not number word (e.g. twenty)
        
        word = word.replace(",","")
        
        # checking first word
        try:
            num = re.search(exp,word).group(0)
        except:
            continue
        
        # change question
        mapping[letter] = float(num)
        new_question = new_question.replace(num, letter, 1)
        letter = chr(ord(letter)+1)
        
    return new_question,mapping
