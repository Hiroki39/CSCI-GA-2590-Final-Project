import pandas as pd
import numpy as np
import math
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
                if('=' in line):
                    equations.append(line)

            answer = evaluate_equations(equations, mapping)
            answers.append(answer)
            equation_column.append(equations)
            if(answer is None):
                print("*****\n",i,response,equations)
                count += 1

    else:
        pass

    print("invalid count", count)
    return answers, equation_column


#if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--logname', type=str, required=True)
    #parser.add_argument('--prompt', type=str, required=True)

filename = "2f6cdbe4-b7bd-43bd-9c67-298cba3f0fb3.jsonl"

result = pd.read_json("logs/"+filename, lines=True)

result['response_answer'], result['equations'] = calculate_answer(result, "arithcot")

result['answer'] = [i[0] for i in result['answer']]

print("acc",np.mean(result['response_answer'] == result['answer']))

result.to_csv("experiment_results/"+ filename + ".csv")

#for i in range(0, len(result)):
#    print("\nsample ",i,"\n")
#    print(result.response[i])
#    print("mapping: ",result.mapping[i])
#    print("answer: ",result.answer[i])