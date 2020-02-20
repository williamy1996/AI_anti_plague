# !/usr/bin/env python
# coding: utf-8
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils.smape import smape
from utils.dataset_loader import load_raw_task_data


def create_submisson_file(pred_test):
    test_data = pd.read_csv('data/candidate_val.csv')
    pred_df = pd.DataFrame(pred_test, columns=['p%d' % i for i in range(1, 7)])
    result = pd.concat([test_data[['id']], pred_df], axis=1).reset_index(drop=True)
    result.to_csv('result.csv', index=False)
    result.head()


X, y = load_raw_task_data()
n_fold = 5
from sklearn import linear_model
reg = linear_model.LinearRegression()
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
