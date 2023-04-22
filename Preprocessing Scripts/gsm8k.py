import re
import pandas as pd
import string

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
        new_question = q
        letter = 'A'

        words = q.split(' ')

        # Finding numbers in the questions

        number_words_count = 0

        for j, word in enumerate(words):

            if(number_words_count):
                number_words_count -= 1
                continue

            try:
                # only support pure numbers, price, and number words

                # checking first word first
                num = None
                num = re.sub(r'[^(%|\w)]','',word)
                num = w2n.word_to_num(num)

                # if the above throws no exception, check whether a multi-number word
                k = j+1
                long_word = re.sub(r'[^(%|\w)]','',word)
                while(k < len(words)):
                    next_word = re.sub(r'[^(%|\w)]','',words[k])
                    try:
                        temp = w2n.word_to_num(next_word)
                        long_word = long_word + " " + next_word
                        k += 1
                        number_words_count += 1
                    except:
                        break

                num = w2n.word_to_num(long_word)
                mapping[letter] = num
                new_question = new_question.replace(long_word, letter, 1)
                letter = chr(ord(letter)+1)
            except:
                continue
        
        if(len(mapping) == 0):
            print(i,q)

        # Finding numbers in the final answer

        ans = None
        try:
            ans = re.findall(r"#### (.*)", a)[0]
            ans = re.sub(r',','',ans)
            ans = float(ans)
        except:
            #print(i, a, ans)
            ans = None

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

dataset.save_to_disk("Data/processed_gsm8k")