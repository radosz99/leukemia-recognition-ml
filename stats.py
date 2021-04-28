import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

scores = np.load('scores.npy')

clfs = ['mlp_100_true', 'mlp_200_true', 'mlp_300_true', 'mlp_100_false', 'mlp_200_false', 'mlp_300_false']

best_features_amount = 8
alfa = .05
t_statistics = [np.zeros((len(clfs), len(clfs))) for i in range (best_features_amount)]
p_values = [np.zeros((len(clfs), len(clfs))) for i in range (best_features_amount)]
best_results = []

for feature_index in range(best_features_amount):
  for clf_id in range(len(clfs)):
    best_results.append(scores[clf_id,feature_index])

for feature_index in range(best_features_amount):
  for i in range(len(clfs)):
    for j in range(len(clfs)):
      t_statistics[feature_index][i][j], p_values[feature_index][i][j] = ttest_ind(best_results[i + feature_index * len(clfs)], best_results[j + feature_index * len(clfs)])

advantages = [np.zeros((len(clfs), len(clfs))) for i in range (best_features_amount)]
for index, t_statistic in enumerate(t_statistics):
  advantages[index][t_statistic > 0] = 1

significances = [np.zeros((len(clfs), len(clfs))) for i in range (best_features_amount)]
for index, p_value in enumerate(p_values):
  significances[index][p_value <= alfa] = 1

stats_better = [significances[i] * advantages[i] for i in range(best_features_amount)]

names_column = np.array([ [header] for header in clfs])
for i in range(best_features_amount):
    current_matrix_table = tabulate(np.concatenate((names_column, stats_better[i]), axis=1), clfs)
    print(f"\nFeature {i+1} - statistically significantly better:\n", current_matrix_table)