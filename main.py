import argparse
import openai
from utils import evaluate_openai
from uuid import uuid4
from dotenv import load_dotenv
import os
import csv


def conduct_test(model_name, dataset_name, prompt, shot, dev):

    run_id = str(uuid4())

    print(f"Run ID: {run_id}")

    # Set up the OpenAI API client
    openai.api_key = os.getenv('OPENAI_API_KEY')
    evaluate_openai(run_id, model_name, dataset_name, prompt, shot, dev)

    if not dev:
        with open('log_files.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([run_id, model_name, dataset_name, prompt, shot])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='gptturbo')
    parser.add_argument('--dataset', type=str, required=True, default='gsm8k')
    parser.add_argument('--prompt', type=str, required=True,
                        default='pycot', choices=['pycot', 'sympy'])
    parser.add_argument('--shot', type=int, required=True,
                        default=1, choices=[1, 2, 4, 8])
    parser.add_argument(
        '--dev', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    print("Current Arguments: ", args)

    load_dotenv()

    conduct_test(args.model, args.dataset, args.prompt, args.shot, args.dev)