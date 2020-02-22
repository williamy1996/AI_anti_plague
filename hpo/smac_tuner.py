import os
import gc
import pickle
import sys
import argparse
import time
import datetime
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import KFold
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

sys.path.append(os.getcwd())
from utils.dataset_loader import load_holdout_data
from utils.smape import smape

# Script:
#   python3 hpo/smac_tuner.py --algo lightgbm --iter_num 5000 --task_id 3 --data_dir data/p1

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
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        cs = KNearestNeighborsRegressor.get_hyperparameter_search_space()
    elif regressor_id == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        cs = LibLinear_SVR.get_hyperparameter_search_space()
    elif regressor_id == 'random_forest':
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 500, default_value=200)
        criterion = CategoricalHyperparameter("criterion",
                                              ['mse', 'friedman_mse', 'mae'])
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0)
        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = \
            UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            'min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")
        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease, bootstrap])

    elif regressor_id == 'lightgbm':
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 20, 100, default_value=31)
        learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
        min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
        reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
        cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                                colsample_bytree, reg_alpha, reg_lambda])
    elif regressor_id == 'catboost':
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
        max_depth = UniformIntegerHyperparameter("max_depth", 4, 10, default_value=6)
        learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
        subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
        colsample_bylevel = UniformFloatHyperparameter("colsample_bylevel", 0.5, 1, default_value=1)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
        loss_function = CategoricalHyperparameter("loss_function", ['RMSE', 'MAE'], default_value='RMSE')
        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, subsample,
                                colsample_bylevel, reg_lambda, loss_function])
    # ---ADD THE HYPERSPACE FOR YOUR REGRESSOR---------------
    else:
        raise ValueError('Undefined regressor identifier: %s!' % regressor_id)
    model = UnParametrizedHyperparameter("estimator", regressor_id)
    cs.add_hyperparameter(model)
    return cs


def get_regressor(_config):
    estimator = _config['estimator']
    config = _config.get_dictionary().copy()
    config.pop('estimator', None)
    config['random_state'] = 1
    if estimator == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        reg = KNearestNeighborsRegressor(**config)
    elif estimator == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        reg = LibLinear_SVR(**config)
    elif estimator == 'random_forest':
        from autosklearn.pipeline.components.regression.random_forest import RandomForest
        reg = RandomForest(**config)
    elif estimator == 'lightgbm':
        from models.lightgbm import LightGBMRegressor
        reg = LightGBMRegressor(**config)
    elif estimator == 'catboost':
        from models.catboost import CatBoostRegressor
        reg = CatBoostRegressor(**config)
    # ---ADD THE CONSTRUCTOR FOR YOUR REGRESSOR---------------
    else:
        raise ValueError('Undefined regressor identifier: %s!' % regressor_id)
    if hasattr(reg, 'n_jobs'):
        setattr(reg, 'n_jobs', -1)
    return reg


# Load data.
# X, y, _ = load_raw_task_data()
X_train, X_valid, y_train, y_valid, _, _ = load_holdout_data(data_dir=data_dir,
                                                             task_id=task_id)


def holdout_evaluation(configuration):
    reg = get_regressor(configuration)
    print(configuration.get_dictionary())
    # Fit this regressor on the train data.
    print('Starting to fit a regression model - %s' % regressor_id)
    _start_time = time.time()
    reg.fit(X_train, y_train)
    score = smape(reg.predict(X_valid), y_valid)
    print('This validation took %.2f seconds.' % (time.time() - _start_time))
    return score


def cv_evaluation(configuration):
    n_fold = 5
    kfold = KFold(n_splits=n_fold, random_state=1, shuffle=True)

    scores = list()
    time_costs = list()

    for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        print('=> Start %d-th validation.' % fold_id)
        _start_time = time.time()
        train_x = X[train_idx]
        valid_x = X[valid_idx]
        train_y = y[train_idx]
        valid_y = y[valid_idx]

        reg = get_regressor(configuration)

        reg.fit(train_x, train_y)
        pred_y = reg.predict(valid_x)

        scores.append(smape(pred_y, valid_y))
        _time_cost = time.time() - _start_time
        print('internal validation took %.2f seconds.' % _time_cost)
        time_costs.append(_time_cost)
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('=> Finishing cv evaluations')
    print(np.mean(scores))
    print(np.mean(time_costs))
    return np.mean(scores)


def obj_func_example(configuration):
    # if evaluation_type == 'holdout':
    #     return holdout_evaluation(configuration)
    # else:
    #     return cv_evaluation(configuration)
    return holdout_evaluation(configuration)


def tune_hyperparamerer_smac():
    configs, results = list(), list()
    config_space = create_hyperspace(regressor_id)
    output_dir = "logs/smac3_output_%s" % (datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f'))
    scenario_dict = {
        'abort_on_first_run_crash': False,
        "run_obj": "quality",
        "cs": config_space,
        "deterministic": "true",
        "runcount-limit": trial_num,
        'output_dir': output_dir
    }
    optimizer = SMAC(scenario=Scenario(scenario_dict),
                     rng=np.random.RandomState(45),
                     tae_runner=obj_func_example)

    optimizer.optimize()

    runhistory = optimizer.solver.runhistory
    runkeys = list(runhistory.data.keys())
    for key in runkeys:
        _val = runhistory.data[key][0]
        _config = runhistory.ids_config[key[0]]
        configs.append(_config.get_dictionary())
        results.append(_val)

    inc_idx = np.argmin(results)
    return configs[inc_idx], (configs, results)


if __name__ == "__main__":
    inc, details = tune_hyperparamerer_smac()
    configs, results = details
    idx = np.argmin(results)
    print('-' * 50)
    print(results[idx])
    print(idx)
    print('Results for all configurations evaluated', results)
    print('The best configuration found is', configs[idx])
    save_path = "smac-%s-%d-%d.pkl" % (regressor_id, trial_num, task_id)
    if not os.path.exists('data'):
        os.mkdir('data')
    with open('data/%s' % save_path, 'wb')as f:
        pickle.dump([configs[idx], configs, results], f)
