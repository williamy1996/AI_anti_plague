# !/usr/bin/env python
# coding: utf-8
import gc
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.path.append(os.getcwd())
from utils.smape import smape
from utils.dataset_loader import load_raw_task_data

parser = argparse.ArgumentParser()
# task_id=0 means aggregate all results on each task[1:6]
# task_id=[1:7) corresponds to the sub-task id.
parser.add_argument('--task_id', type=str, default='3')


def execute_task(task_id):
    # Load raw data for some task.
    X, y = load_raw_task_data(task_id=task_id)
    from sklearn import linear_model
    # You can set hyperparameter here.
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

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
