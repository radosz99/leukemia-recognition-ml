import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.feature_selection import SelectKBest, f_classif

classes = {}
symptoms = []
file_prefix = 'output/bialaczka_'
classes_size = 0
data_list = []


def get_lines_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as myfile:
        return myfile.readlines()


def read_from_file_to_dict(filename):
    return dict(enumerate(read_from_file_to_list(filename), start=1))


def read_from_file_to_list(filename):
    return [line.rstrip('\n') for line in get_lines_from_file(filename)]


def read_parameters_from_file():
    global classes, symptoms
    classes = read_from_file_to_dict('jednostki.txt')
    symptoms = read_from_file_to_list('cechy.txt')


def parse_csv(i, filename):
    with open(f"{filename}{i}.txt", 'r') as myfile:
        data_set = pd.read_csv(myfile, sep=' ', header=None)
        data_set['Nr klasy'] = i
        return data_set


def evaluate():
    data_list = [parse_csv(i, file_prefix) for i in range(1, classes_size + 1)]
    dataset = pd.concat(data_list, axis=0)
    dataset.columns = symptoms
    return create_ranking(dataset)


def create_ranking(dataset):
    x = dataset.drop('Nr klasy', axis=1)
    y = dataset['Nr klasy']
    k_best_selector = SelectKBest(score_func=f_classif, k=len(symptoms) - 1)
    k_best_selector.fit(x, y)
    scores_ranking = [(symptom, round(score, 2)) for symptom, score in zip(x.columns, k_best_selector.scores_)]
    scores_ranking.sort(reverse=True, key=lambda x: x[1])
    return scores_ranking


def divide_data_to_files():
    block_counter = 0
    if(not os.path.isdir('output')):
        os.mkdir('output')
    with open('bialaczka.csv', 'r') as myfile:
        block_lines = []
        for line in myfile.readlines():
            index = line.find(';')
            if(index > 0):  # new block
                if(block_counter > 0):  # to avoid empty first block
                    with open(f"{file_prefix}{block_counter}.txt", 'w') as resultfile:
                        resultfile.writelines(block_lines)
                    block_lines = []
                block_counter += 1
            line = line[index + 1:]
            index = line.find(';')
            line = line[index + 1:]
            line = line.replace(';', ' ')
            block_lines.append(line)
        with open(f"{file_prefix}{block_counter}.txt", 'w') as resultfile:  # to get last block
            resultfile.writelines(block_lines)
    global classes_size
    classes_size = block_counter


def start():
    read_parameters_from_file()
    divide_data_to_files()
    ranking = evaluate()
    for item in ranking:
        print(f"{item[0]} - {item[1]}")


if(__name__ == '__main__'):
    start()
