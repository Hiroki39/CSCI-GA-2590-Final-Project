import pandas as pd
import numpy as np
import math, re
from collections import OrderedDict

import argparse
import os

def evaluate_equations(equations, mapping):
    
    new_variables = OrderedDict()
    ceil = lambda x: math.ceil(x)
    floor = lambda x: math.floor(x)
    try:
        # Extract variables from mapping
        for i in mapping:
            value = mapping[i]
            exec(i + ' = ' + str(value))

        # Evaluate equation one by one
        for equation in equations:
            elements = equation.split('=')
            name = elements[0].strip().replace(' ', '_')
            expression = elements[1].strip()

            value = eval(expression)

            exec(name + ' = ' + str(value))
            new_variables[name] = value

    except Exception as e:
        print(e)
        return None
    
    ret = list(new_variables.items())
    try:
        return ret[-1][1]
    except:
        return None
       
def extract_answer(response, prompt = 'cot'):

    answer = None
    if(prompt == 'cot'):
        try:
            answer = re.findall(r'(?<=The answer is ).+', response)[0]
        except:
            return None
        
        try:
            answer = re.search(r'\d+(\,\d+)*(\.\d+)?', answer).group(0)
        except:
            return None
    elif(prompt == 'zero-cot'):

        # Use the number in the response
        try:
            answer = re.findall(r'\d+(?:\.\d+)?', response)[-1]
        except:
            return None
        
    return float(answer)
    
def calculate_answer(result, prompt):

    answers = []
    equation_column = []
    count = 0

    if(prompt == 'arithcot'):

        for i in range(len(result)):
            response = result.response[i]
            mapping = result.mapping[i]

            equations = []

            # Extract eqautions from response
            for line in response.split('\n'):
                if('=' in line and not line[0].isdigit()):
                    equations.append(line)

            answer = evaluate_equations(equations, mapping)
            answers.append(answer)
            equation_column.append(equations)
            if(answer is None):
                print("*****\n",i,response,equations)
                count += 1

    elif(prompt == 'cot' or prompt == 'zero-cot'):

        for i in range(len(result)):
            response = result.response[i]
            
            answer = extract_answer(response, prompt)
            answers.append(answer)
            if(answer is None):
                print("*****\n",i,response)
                count += 1

    else:
        pass

    print("invalid count", count)
    return answers, equation_column

# evaluate aqua dataset with multiple choice
def eval_aqua(result):
    # output = pd.read_json("logs/e847e842-087c-4ff9-ac07-55ee02041327.jsonl", lines=True)
    total, correct, undef = len(result), 0, 0
    for i, response in enumerate(result['response']):
        a = re.search(r"[Aa]nswer: [ABCDE]", response)
        if a:
            _, end = a.span()
            predicted = response[end-1]
            if predicted == result['answer'][i]:
                correct += 1
        else:
            print(response)
            undef += 1
    print("acc", correct/total, "invalid", undef/total)




# kinda too messy; needs to be cleaned
def eval_result(filename, prompt, dataset_name):
    
    result = pd.read_json("logs/"+filename+".jsonl", lines=True)

    if prompt == 'arithcot':
        result['response_answer'], result['equations'] = calculate_answer(result, "arithcot")
        result['answer'] = [i[0] for i in result['answer']]
        print("acc",np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'cot' or prompt == 'zero-cot':
        if(prompt == 'zero-cot'):
            raise Warning("Refer to the paper for zero-cot evaluation")
        result['response_answer'],  _ = calculate_answer(result, prompt)
        result['answer'] = [i[0] for i in result['answer']]
        print("acc",np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'sympy':
        eval_aqua(result)
    else:
        print("eval not implemented")
        pass
    
    result.to_csv("experiment_results/"+ filename + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logname', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    eval_result(args.logname, args.prompt, args.dataset)

#for i in range(0, len(result)):
#    print("\nsample ",i,"\n")
#    print(result.response[i])
#    print("mapping: ",result.mapping[i])
#    print("answer: ",result.answer[i])