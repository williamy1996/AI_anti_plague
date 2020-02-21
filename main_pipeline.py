# !/usr/bin/env python
# coding: utf-8
import gc
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
from utils.smape import smape
from utils.dataset_loader import load_raw_task_data
from utils.file_operator import save_task_result, create_submission_file

parser = argparse.ArgumentParser()
# task_id=0 means aggregate all results on each task[1:6]
# task_id=[1:7) corresponds to the sub-task id.
parser.add_argument('--task_ids', type=str, default='3')
parser.add_argument('--method_id', type=str, default='lr')


def execute_task(method_id: str, task_id: int):
    # Load raw data for some task.
    X, y, X_test = load_raw_task_data(task_id=task_id)
    from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    # config = {'C': 7069.872331099711,
    #           'dual': False,
    #           'epsilon': 0.0010554244565115106,
    #           'fit_intercept': True,
    #           'intercept_scaling': 1,
    #           'loss': 'squared_epsilon_insensitive',
    #           'tol': 6.305030629525869e-05
    #           }
    # reg = LibLinear_SVR(**config)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    save_task_result(method_id, task_id, y_pred)


if __name__ == "__main__":
    args = parser.parse_args()
    method_id = args.method_id
    task_ids = [int(task_id) for task_id in args.task_ids.split(',')]
    for task_id in task_ids:
        if task_id != 0:
            execute_task(method_id, task_id)
        else:
            create_submission_file(method_id)
