import pandas as pd

import argparse
import openai
from utils import evaluate_openai
from uuid import uuid4
from dotenv import load_dotenv
import os
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, default='gptturbo')

    result = pd.read_json(parser.run, lines=True)

    for i in range(0, len(result)):
        print("\nsample ",i,"\n")
        print(result.response[i])
        print("mapping: ",result.mapping[i])
        print("answer: ",result.answer[i])