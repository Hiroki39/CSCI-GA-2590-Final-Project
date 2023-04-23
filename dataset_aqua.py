import re
from utils import extract_mapping

from datasets import load_dataset

# Loading the dataset

dataset = load_dataset("aqua_rat", "raw")

# Function for extracting mapping and answer from quesitons

def extract_mapping_only(dataset):

    train_mappings = []
    train_new_questions = []

    failed = []

    # Replacing numbers with letters

    for i, d in enumerate(dataset):

        q = d['question']
        
        new_question, mapping = extract_mapping(q)

        if (len(mapping) == 0):
            failed.append((i,q))

        # Finding numbers in the final answe

        train_mappings.append(mapping)
        train_new_questions.append(new_question)

    # Adding new columns to train

    dataset = dataset.add_column("mapping", train_mappings)
    dataset = dataset.add_column("new_question", train_new_questions)

    return dataset, failed

dataset['train'], train_failed = extract_mapping_only(dataset['train'])
dataset['test'], test_failed = extract_mapping_only(dataset['test'])
