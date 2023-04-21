import re
import pandas as pd

from word2number import w2n

from datasets import load_dataset

# Loading the dataset

dataset = load_dataset("gsm8k", "main")

# Function for extracting mapping and answer from quesitons

def extract_mapping_ans(dataset):

    train_mappings = []
    train_new_questions = []
    train_num_answers = []

    # Replacing numbers with letters

    for i,d in enumerate(dataset):

        q = d['question']
        a = d['answer']
        mapping = {}
        new_question = []
        letter = 'A'

        words = q.split(' ')

        # Finding numbers in the questions

        for j, word in enumerate(words):

            try:
                # only support pure numbers
                #num = re.sub(r'[^(%|\w)]','',word)
                #num = w2n.word_to_num(num)

                num = re.sub(r'$','',word)
                num = float(num)

                if(num == None):
                    raise ValueError("None num")

                mapping[letter] = num
                new_question.append(letter)
                letter = chr(ord(letter)+1)
            except:
                new_question.append(word)
                continue
        
        #if(len(mapping) == 0):
            #print(i,q)

        # Finding numbers in the final answer

        ans = None
        try:
            ans = re.findall(r"#### (.*)", a)[0]
            ans = re.sub(r',','',ans)
            ans = float(ans)
        except:
            print(i, a, ans)
            ans = None
        
        new_question = ' '.join(new_question)

        train_mappings.append(mapping)
        train_new_questions.append(new_question)
        train_num_answers.append(ans)

    # Adding new columns to train

    dataset = dataset.add_column("mapping", train_mappings)
    dataset = dataset.add_column("new_question", train_new_questions)
    dataset = dataset.add_column("num_answer", train_num_answers)

    return dataset

dataset['train'] = extract_mapping_ans(dataset['train'])
dataset['test'] = extract_mapping_ans(dataset['test'])

dataset.save_to_disk("processed_gsm8k.dat")