import re
import pandas as pd
import string

from datasets import load_dataset

# Loading the dataset

dataset = load_dataset("gsm8k", "main")

# Function for extracting mapping and answer from quesitons

def extract_mapping_ans(dataset):
    
    count = 0

    train_mappings = []
    train_new_questions = []
    train_num_answers = []

    # Replacing numbers with letters

    for i,d in enumerate(dataset):
        
        q = d['question']
        a = d['answer']
        
     # Expression used to extract number
        
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