import argparse
import openai
from datasets import load_dataset
from utils import evaluate_openai
from uuid import uuid4
from dotenv import load_dotenv
import os
import csv


def conduct_test(model, dataset_name):

    run_id = str(uuid4())

    print(f"Run ID: {run_id}")

    # Load the GSM8K dataset from Hugging Face
    if dataset_name == 'gsm8k':
        dataset = load_dataset(dataset_name, 'main')
    # Set up the OpenAI API client
    openai.api_key = os.getenv('OPENAI_API_KEY')
    evaluate_openai(run_id, model, dataset)

    with open('log_files.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model, dataset_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='gptturbo')
    parser.add_argument('--dataset', type=str, required=True, default='gsm8k')
    args = parser.parse_args()

    print("Current Arguments: ", args)

    load_dotenv()

    conduct_test(args.model, args.dataset)
