import re
from utils import extract_mapping

from datasets import load_dataset

# Loading the dataset

dataset = load_dataset("gsm8k", "main")

# Function for extracting mapping and answer from quesitons

def extract_mapping_ans(dataset):

    train_mappings = []
    train_new_questions = []
    train_num_answers = []

    failed = []

    # Replacing numbers with letters

    for i, d in enumerate(dataset):

        q = d['question']
        a = d['answer']
        
        new_question, mapping = extract_mapping(q)

        if (len(mapping) == 0):
            failed.append((i,q))

        # Finding numbers in the final answer

        ans = None
        try:
            ans = re.findall(r"#### (.*)", a)[0]
            ans = re.sub(r',', '', ans)
            ans = float(ans)
        except:
            ans = None

        train_mappings.append(mapping)
        train_new_questions.append(new_question)
        train_num_answers.append(ans)

    # Adding new columns to train

    dataset = dataset.add_column("mapping", train_mappings)
    dataset = dataset.add_column("new_question", train_new_questions)
    dataset = dataset.add_column("num_answer", train_num_answers)

    return dataset, failed

dataset1, test_failed = extract_mapping_ans(dataset['test'])

dataset1.save_to_disk("data/gsm8k")
