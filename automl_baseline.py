# !/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse
import multiprocessing
sys.path.append(os.getcwd())
from utils.dataset_loader import load_raw_task_data
from utils.file_operator import save_task_result, create_submission_file
from utils.smape import smape

parser = argparse.ArgumentParser()
parser.add_argument('--task_ids', type=str, default='1,2,3,4,5,6,0')
parser.add_argument('--method_id', type=str, default='ausk')
parser.add_argument('--time_cost', type=int, default=3600)

import autosklearn.metrics
smape_error = autosklearn.metrics.make_scorer('smape', smape, optimum=0, greater_is_better=False)


def execute_task(time_cost: int, method_id: str, task_id: int):
    # Load raw data for some task.
    X, y, X_test = load_raw_task_data(task_id=task_id)
    import autosklearn.regression
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=time_cost,
        per_run_time_limit=600,
        include_estimators=['random_forest', 'gradient_boosting', 'adaboost', 'extra_trees'],
        n_jobs=multiprocessing.cpu_count()-1,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.7}
    )
    print(automl)
    automl.fit(X, y, dataset_name='anti_plague', metric=smape_error)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    save_task_result(method_id, task_id, predictions)


if __name__ == "__main__":
    args = parser.parse_args()
    method_id = args.method_id
    task_ids = [int(task_id) for task_id in args.task_ids.split(',')]
    time_cost = args.time_cost
    for task_id in task_ids:
        if task_id != 0:
            execute_task(time_cost, method_id, task_id)
        else:
            create_submission_file(method_id)
