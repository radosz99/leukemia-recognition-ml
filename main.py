import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from parse import get_lines_from_file, read_from_file_to_dict, read_from_file_to_list

clfs = {
    'mlp_100_true': MLPClassifier(max_iter=10000, hidden_layer_sizes=100, solver='sgd', momentum=0.9),
    'mlp_200_true':MLPClassifier(max_iter=10000, hidden_layer_sizes=200, solver='sgd', momentum=0.9),
    'mlp_300_true': MLPClassifier(max_iter=10000, hidden_layer_sizes=300, solver='sgd', momentum=0.9),
    'mlp_100_false': MLPClassifier(max_iter=10000, hidden_layer_sizes=100, solver='sgd', momentum=0),
    'mlp_200_false': MLPClassifier(max_iter=10000, hidden_layer_sizes=200, solver='sgd', momentum=0),
    'mlp_300_false': MLPClassifier(max_iter=10000, hidden_layer_sizes=300, solver='sgd', momentum=0)
}

classes, features = {}, []
classes_amount, features_amount, best_features_amount = None, None, 8
X, y = None, None

split_amount, repeat_amount = 2, 5

scores = np.zeros((len(clfs), best_features_amount, split_amount * repeat_amount))
rskf = RepeatedKFold(
    n_splits=split_amount, n_repeats=repeat_amount, random_state=42)

file_prefix = 'output/bialaczka_'


def evaluate():
    data_list = [parse_csv(i, file_prefix) for i in range(1, classes_amount + 1)]
    dataset = pd.concat(data_list, axis=0)
    dataset.columns = features
    return create_ranking(dataset)


def parse_csv(i, filename):
    with open(f"{filename}{i}.txt", 'r') as myfile:
        data_set = pd.read_csv(myfile, sep=' ', header=None)
        data_set['Nr klasy'] = i
        return data_set

def read_parameters_from_file():
    global classes, features, classes_amount, features_amount
    classes = read_from_file_to_dict('jednostki.txt')
    features = read_from_file_to_list('cechy.txt')
    classes_amount = len(classes)
    features_amount = len(features)

def create_ranking(dataset):
    global X, y
    X = dataset.drop('Nr klasy', axis=1)
    y = dataset['Nr klasy']
    k_best_selector = SelectKBest(score_func=f_classif, k=len(features) - 1)
    k_best_selector.fit(X, y)
    scores_ranking = [(symptom, round(score, 2)) for symptom, score in zip(X.columns, k_best_selector.scores_)]
    scores_ranking.sort(reverse=True, key=lambda x: x[1])
    return scores_ranking


def calculate():
    for features_index in range(0, best_features_amount):
        k_best_selector = SelectKBest(score_func=f_classif, k=features_index + 1)
        selected_data = k_best_selector.fit_transform(X, y)
        for fold_id, (train, test) in enumerate(rskf.split(selected_data, y)):
            for clf_id, clf_name in enumerate(clfs):
                X_train, X_test = selected_data[train], selected_data[test]
                y_train, y_test = y.iloc[train], y.iloc[test]
                clf = clone(clfs[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores[clf_id, features_index, fold_id] = score
                print(score)
    print(scores)
    np.save('scores', scores)

if(__name__ == '__main__'):
    read_parameters_from_file()
    ranking = evaluate()
    # for item in ranking:
    #     print(f"{item[0]} - {item[1]}")
    calculate()