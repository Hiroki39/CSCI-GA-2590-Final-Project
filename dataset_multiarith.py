import pandas as pd
from utils import extract_mapping

from datasets import Dataset

# Loading the dataset

df = pd.read_json("data/MultiArith.json")
df.columns = ['index', 'alignments', 'equation', 'answer', 'question']
dataset = Dataset.from_pandas(df)

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
            failed.append((i, q))

        train_mappings.append(mapping)
        train_new_questions.append(new_question)

    # Adding new columns to the dataset

    dataset = dataset.add_column("mapping", train_mappings)
    dataset = dataset.add_column("new_question", train_new_questions)

    return dataset, failed


dataset_processed, train_failed = extract_mapping_only(dataset)
dataset_processed.save_to_disk("data/multiarith")
