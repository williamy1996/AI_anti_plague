# !/usr/bin/env python
# coding: utf-8
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils.smape import smape
from utils.dataset_loader import load_raw_task_data


def save_task_result(task_id, pred_y):
    np.save('data/pred_results_task-%d.npy' % task_id, pred_y)


def create_submission_file():
    pred_test = list()
    for task_id in range(1, 7):
        _pred = np.load('data/pred_results_task%d.npy' % task_id)
        pred_test.append(_pred)
    pred_test = np.array(pred_test).transpose()
    test_data = pd.read_csv('data/candidate_val.csv')
    pred_df = pd.DataFrame(pred_test, columns=['p%d' % i for i in range(1, 7)])
    result = pd.concat([test_data[['id']], pred_df], axis=1).reset_index(drop=True)
    result.to_csv('data/result.csv', index=False)
    print(result.head())


# Load raw data for some task.
X, y = load_raw_task_data(task_id=3)

# KFold or holdout is okay.
n_fold = 5
kfold = KFold(n_splits=n_fold, random_state=1, shuffle=True)

scores = list()
time_costs = list()

for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
    print('Start %d-th validation.' % fold_id)
    _start_time = time.time()
    train_x = X[train_idx]
    valid_x = X[valid_idx]
    train_y = y[train_idx]
    valid_y = y[valid_idx]

    from sklearn import linear_model
    # You can set hyperparameter here.
    reg = linear_model.LinearRegression()

    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)

    scores.append(smape(pred_y, valid_y))
    _time_cost = time.time() - _start_time
    print('This validation took %.2f seconds.' % _time_cost)
    time_costs.append(_time_cost)
    del train_x, train_y, valid_x, valid_y
    gc.collect()

print(np.mean(scores))
print(np.mean(time_costs))

# Inference Process.
# 1. refit the model on whole dataset.
# 2. predict the test data.
# 3. for each task, save the result to file: `save_task_result(task_id, pred_y)`.
# 4. finally, create the final submission file: `create_submission_file`.
