import pandas as pd
import numpy as np
import math
import re
from collections import OrderedDict

import argparse
import os
import sys
from io import StringIO
import contextlib

import random

def evaluate_equations(equations, mapping, response=''):

    # helper
    def isvalid(word):
        if (len(word) <= 1):
            return True
        elif (word[0] == 'c' and word[1].isdigit()):
            return True
        elif ("(" in word or ")" in word):
            return True
        elif (word[0:5] == 'ceil('
              or word[0:6] == 'floor('
              or word[0:6] == 'round('):
            return True

    new_variables = OrderedDict()
    def ceil(x): return math.ceil(x)
    def floor(x): return math.floor(x)
    # Extract variables from mapping
    for i in mapping:
        value = mapping[i]
        exec(i + ' = ' + str(value))
    # Evaluate equation one by one
    for equation in equations:
        elements = equation.split('=')

        # left side
        name = elements[0].strip().replace(' ', '_')
        # right side
        try:
            expression = elements[1].strip()

            expression = expression.replace(' x ', ' * ')
        except:
            print(equation)
            continue

        # normal evaluation
        if (expression[-1] == '.'):
            expression = expression[:-1]

        for variable in mapping:
            expression = expression.replace(variable, str(mapping[variable]))

        for variable in new_variables:
            expression = expression.replace(
                variable, str(new_variables[variable]))

        expression = re.sub(r'[a-z]', '', expression)
        expression = re.sub(r'[A-Z]', '', expression)

        try:
            value = eval(expression)
            exec(name + ' = ' + str(value))
        except Exception as e:
            print('1')
            print(expression)
            continue
        new_variables[name] = value

    if 'answer' in new_variables:
        return new_variables['answer']
    else:
        try:
            equation2 = equations[-2]
            exec(equation2)
            equation1 = equations[-1]
            exec(equation1)
            return answer
        except Exception as e:
            print('!', e)
            print(response)
            if (len(new_variables) >= 1):
                return list(new_variables.values())[-1]
            else:
                return None


def evaluate_function(mapping, response):
    # print("="*10)
    # print(response.split('\n')[0])
    try:
        exec(response)
    except:
        try:
            response = response[5:].strip()
            start, _ = re.search('def', response).span()
            response = response[start:]
            exec(response)
        except:
            print('****** Fn Problem')
            print(response)
            return None

    arguments = ''
    for name in mapping:
        exec(name + '=' + str(mapping[name]))
        arguments = arguments + name + ','
    arguments = arguments[:-1]

    try:
        problem = response.split('\n')[0].split()[1].split('(')[0]
        answer = eval(problem + '('+arguments+')')
        return answer
    except Exception as e:
        print(e)
        print('****** Exec Problem')
        return None


def extract_answer(response, prompt='cot'):

    answer = None
    flag = 0
    if (prompt == 'cot'):
        try:
            answer = re.findall(r'(?<=The answer is ).+', response)[0]
            answer = re.search(r'\d+(\,\d+)*(\.\d+)?', answer).group(0)
            # to be tidied
            try:
                if (re.search(r'\d+(\,\d+)*(\.\d+)?', answer).group(0)
                   != re.search(r'\d+(\,\d+)*(\.\d+)?', answer).group(-1)):
                    flag = 1
            except:
                pass
        except:
            try:
                answer = re.findall(r'\d+(?:\.\d+)?', response)[-1]
                flag = 1
            except:
                return None, 1
    elif (prompt == 'zero-cot'):

        # Use the number in the response
        try:
            answer = re.findall(r'\d+(?:\.\d+)?', response)[-1]
        except:
            return None

    return float(answer.replace(",", '')), flag


def calculate_answer(result, prompt):

    answers = []
    equation_column = []
    count = 0

    if (prompt == 'arithcot'):

        for i in range(len(result)):
            response = result.response[i]
            mapping = result.mapping[i]

            equations = []

            # Extract eqautions from response
            for line in response.split('\n'):
                if ('=' in line and not line[0].isdigit()):
                    equations.append(line)

            answer = evaluate_equations(equations, mapping)
            answers.append(answer)
            equation_column.append(equations)
            if (answer is None):
                print("*****\n", i, response, equations)
                count += 1

    elif (prompt == 'cot' or prompt == 'zero-cot'):

        for i in range(len(result)):
            response = result.response[i]

            answer, flag = extract_answer(response, prompt)
            equation_column.append(flag)
            answers.append(answer)
            if (answer is None):
                print("*****\n", i, response)
                count += 1

    elif (prompt == 'varcot'):

        for i in range(len(result)):
            response = result.response[i]
            mapping = result.mapping[i]

            equations = []

            # Extract eqautions from response
            for line in response.split('\n'):
                if ('=' in line and not line[0].isdigit()):
                    equation = line.split("=")
                    left = equation[0].strip().split(" ")
                    left = left[-1]
                    right = equation[1]
                    equation = left + " = " + right
                    equations.append(equation)
                elif ("The answer is" in line):
                    newline = line.split('.')[0]
                    newline = newline.replace("The answer is", "answer =")
                    equations.append(newline.strip())

            answer = evaluate_equations(equations, mapping, response)
            answers.append(answer)
            equation_column.append(equations)
            if (answer is None):
                print("*****\n", i, response, equations)
                count += 1
    elif (prompt == 'pycot'):

        for i in range(len(result)):
            response = result.response[i]
            mapping = result.mapping[i]

            answer = evaluate_function(mapping, response)
            answers.append(answer)
            if (answer is None):
                print("*****\n", i, response)
                count += 1

    else:
        pass

    print("invalid count", count)
    return answers, equation_column

# evaluate aqua dataset with multiple choice
def eval_aqua(result):
    total, correct, undef = len(result), 0, 0
    for i, response in enumerate(result['response']):
        a = re.search(
            r"([Aa]nswer|[Cc]hoice|[Oo]ption) (?:is )?[A-Ea-e]", response)
        if a:
            _, end = a.span()
            predicted = response[end-1]
            if predicted.lower() == result['answer'][i].lower():
                correct += 1
            else:
                pass
                # print("="*10)
                # print('correct answer: ', result['answer'][i])
                # print(result['question'][i], response)
        else:
            print(result['question'][i])
            print(response)
            print("="*10)
            undef += 1
    print("acc", correct/total, "invalid", undef/total)

# for execute print function 
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

# extract sympy code from output, then exec the codes
def extract_equations(response):
    ls = response.split("\n")
    start = None
    for ind, item in enumerate(ls):
        if item and item.strip()[0]== "#":
            start = ind+1
            break
    if start is None:
        return None
    code = "\n".join(ls[start:-1])
    with stdoutIO() as s:
        try:
            exec(code)
        except Exception as e:
            print(e)
            # print(response)
            return None
    try:
        res = float(s.getvalue().strip()[1:-1])
    except Exception as e:
        print(e)
        # print(response)
        return None
    return [res]

def eval_sympy(result):
    answers = []
    invalid = 0
    for i in range(len(result)):
        answer = extract_equations(result['response'][i])
        answers.append(answer)
        invalid += 1 if answer is None else 0
    print("invalid", invalid)
    return answers

# kinda too messy; needs to be cleaned

def eval_result(filename, prompt, dataset_name):

    result = pd.read_json("logs/" + filename + ".jsonl", lines=True)

    if prompt == 'arithcot':
        result['response_answer'], result['equations'] = calculate_answer(
            result, "arithcot")
        result['answer'] = [i[0] for i in result['answer']]
        print("acc", np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'cot' or prompt == 'zero-cot':
        if (prompt == 'zero-cot'):
            raise Warning("Refer to the paper for zero-cot evaluation")
        result['response_answer'], result['flag'] = calculate_answer(
            result, 'cot')
        result['answer'] = [i[0] for i in result['answer']]
        print("acc", np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'varcot':
        result['response_answer'], result['equations'] = calculate_answer(
            result, "varcot")
        result['answer'] = [i[0] for i in result['answer']]
        print("acc", np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'pycot':
        result['response_answer'], _ = calculate_answer(
            result, "pycot")
        result['answer'] = [i[0] for i in result['answer']]
        print("acc", np.mean(result['response_answer'] == result['answer']))
    elif prompt == 'sympy':
        if dataset_name == 'aqua_rat':
            eval_aqua(result)
        else:
            result['response_answer'] = eval_sympy(result)
            print("acc", np.mean(result['response_answer'] == result['answer']))
    else:
        print("eval not implemented")
        pass

    result.to_csv("experiment_results/" + filename + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logname', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    eval_result(args.logname, args.prompt, args.dataset)

# for i in range(0, len(result)):
#    print("\nsample ",i,"\n")
#    print(result.response[i])
#    print("mapping: ",result.mapping[i])
#    print("answer: ",result.answer[i])
