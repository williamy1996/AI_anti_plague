import os
import sys
import pickle
import time
import argparse
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

sys.path.append(os.getcwd())

from utils.dataset_loader import load_holdout_data
from utils.smape import smape

# Script:
#   python3 hpo/tpe_tuner.py --algo lightgbm --iter_num 5000 --task_id 3 --data_dir data/p1

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='lightgbm', choices=['lightgbm', 'random_forest', 'catboost'])
parser.add_argument('--iter_num', type=int, default=500)
parser.add_argument('--task_id', type=int, default=3)
parser.add_argument('--data_dir', type=str, default='data/')
args = parser.parse_args()

regressor_id = args.algo
trial_num = args.iter_num
task_id = args.task_id
data_dir = args.data_dir


def create_hyperspace(regressor_id):
    if regressor_id == 'knn':
        cs = {'n_neighbors': hp.randint('knn_n_neighbors', 100) + 1,
              'weights': hp.choice('knn_weights', ['uniform', 'distance']),
              'p': hp.choice('knn_p', [1, 2])}
    elif regressor_id == 'liblinear_svr':
        cs = {'loss': hp.choice('liblinear_combination', [{'loss': "epsilon_insensitive", 'dual': "True"},
                                                          {'loss': "squared_epsilon_insensitive", 'dual': "True"},
                                                          {'loss': "squared_epsilon_insensitive", 'dual': "False"}]),
              'dual': None,
              'epsilon': hp.loguniform('liblinear_epsilon', np.log(1e-3), np.log(1)),
              'tol': hp.loguniform('liblinear_tol', np.log(1e-5), np.log(1e-1)),
              'C': hp.loguniform('liblinear_C', np.log(0.03125), np.log(32768)),
              'fit_intercept': hp.choice('liblinear_fit_intercept', ["True"]),
              'intercept_scaling': hp.choice('liblinear_intercept_scaling', [1])}
    elif regressor_id == 'random_forest':
        cs = {'n_estimators': hp.choice('rf_n_estimators', [100]),
              'criterion': hp.choice('rf_criterion', ["mse", "mae"]),
              'max_features': hp.uniform('rf_max_features', 0, 1),
              'max_depth': hp.choice('rf_max_depth', [None]),
              'min_samples_split': hp.randint('rf_min_samples_split', 19) + 2,
              'min_samples_leaf': hp.randint('rf_min_samples_leaf', 20) + 1,
              'min_weight_fraction_leaf': hp.choice('rf_min_weight_fraction_leaf', [0]),
              'max_leaf_nodes': hp.choice('rf_max_leaf_nodes', [None]),
              'min_impurity_decrease': hp.choice('rf_min_impurity_decrease', [0]),
              'bootstrap': hp.choice('rf_bootstrap', ["True", "False"])}
    elif regressor_id == 'lightgbm':
        cs = {'n_estimators': hp.randint('lgb_n_estimators', 901) + 100,
              'num_leaves': hp.randint('lgb_num_leaves', 81) + 20,
              'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.025), np.log(0.3)),
              'min_child_weight': hp.randint('lgb_min_child_weight', 10) + 1,
              'subsample': hp.uniform('lgb_subsample', 0.5, 1),
              'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.5, 1),
              'reg_alpha': hp.loguniform('lgb_reg_alpha', np.log(1e-10), np.log(10)),
              'reg_lambda': hp.loguniform('lgb_reg_lambda', np.log(1e-10), np.log(10))
              }
    elif regressor_id == 'your_regressor_id':
        pass
    # ---ADD THE HYPERSPACE FOR YOUR REGRESSOR---------------
    else:
        raise ValueError('Undefined regressor identifier: %s!' % regressor_id)
    cs['estimator'] = regressor_id
    return cs


def get_regressor(_config):
    estimator = _config['estimator']
    config = _config.copy()
    config.pop('estimator', None)
    if estimator == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        reg = KNearestNeighborsRegressor(**config)
    elif estimator == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        nested = config['loss']
        config['dual'] = nested['dual']
        config['loss'] = nested['loss']
        reg = LibLinear_SVR(**config)
    elif estimator == 'random_forest':
        from autosklearn.pipeline.components.regression.random_forest import RandomForest
        reg = RandomForest(**config)
    elif estimator == 'lightgbm':
        from models.lightgbm import LightGBMRegressor
        reg = LightGBMRegressor(**config)
    elif estimator == 'your_regressor_id':
        # Based on the hyperparameter configuration `config`, construct the regressor.
        pass
    # ---ADD THE CONSTRUCTOR FOR YOUR REGRESSOR---------------
    else:
        raise ValueError('Undefined regressor identifier: %s!' % regressor_id)
    if hasattr(reg, 'n_jobs'):
        setattr(reg, 'n_jobs', -1)
    return reg


# Load data.
X_train, X_valid, y_train, y_valid, _, _ = load_holdout_data(data_dir=data_dir,
                                                             task_id=task_id)


def holdout_evaluation(configuration):
    print(configuration)
    reg = get_regressor(configuration)

    # Fit this regressor on the train data.
    print('Starting to fit a regression model - %s' % regressor_id)
    _start_time = time.time()
    reg.fit(X_train, y_train)
    score = smape(reg.predict(X_valid), y_valid)
    print('This validation took %.2f seconds.' % (time.time() - _start_time))
    return score


def tune_hyperparameter_tpe():
    configs, results = list(), list()

    def objective(x):
        return {
            'loss': holdout_evaluation(x),
            'status': STATUS_OK,
            'config': x
        }

    config_space = create_hyperspace(regressor_id)
    trials = Trials()
    fmin(objective, config_space, tpe.suggest, trial_num, trials=trials)

    # output_dir = "logs/hyperopt_output_%s" % (datetime.datetime.fromtimestamp(
    #     time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))

    for trial in trials.trials:
        config = trial['result']['config']
        perf = trial['result']['loss']
        configs.append(config)
        results.append(perf)

    inc_idx = np.argmin(results)
    return configs[inc_idx], (configs, results)


if __name__ == "__main__":
    inc, details = tune_hyperparameter_tpe()
    configs, results = details
    idx = np.argmin(results)
    print('-' * 50)
    print(results[idx])
    print(idx)
    print('Results for all configurations evaluated', results)
    print('The best configuration found is', configs[idx])
    save_path = "hyperopt-%s-%d-%d.pkl" % (regressor_id, trial_num, task_id)
    if not os.path.exists('data'):
        os.mkdir('data')
    with open('data/%s' % save_path, 'wb')as f:
        pickle.dump([configs[idx], results[idx]])
