import os
import gc
import sys
import time
import datetime
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import train_test_split
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from sklearn.model_selection import KFold

sys.path.append(os.getcwd())
from utils.dataset_loader import load_raw_task_data
from utils.smape import smape

# ---CHANGE SETTINGS HERE---------------
regressor_id = 'random_forest'
trial_num = 50
# evaluation_type = ['holdout', 'cv']
evaluation_type = 'holdout'
# -----------------------------


def create_hyperspace(regressor_id):
    if regressor_id == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        cs = KNearestNeighborsRegressor.get_hyperparameter_search_space()
    elif regressor_id == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        cs = LibLinear_SVR.get_hyperparameter_search_space()
    elif regressor_id == 'random_forest':
        from autosklearn.pipeline.components.regression.random_forest import RandomForest
        cs = RandomForest.get_hyperparameter_search_space()
    elif regressor_id == 'your_regressor_id':
        cs = ConfigurationSpace()
        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])
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
    if estimator == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        reg = KNearestNeighborsRegressor(**config)
    elif estimator == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        reg = LibLinear_SVR(**config)
    elif estimator == 'random_forest':
        from autosklearn.pipeline.components.regression.random_forest import RandomForest
        reg = RandomForest(**config)
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
X, y = load_raw_task_data()


def holdout_evaluation(configuration):
    reg = get_regressor(configuration)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        shuffle=True, random_state=1)
    # Fit this regressor on the train data.
    print('Starting to fit a regression model - %s' % regressor_id)
    _start_time = time.time()
    reg.fit(X_train, y_train)
    score = smape(reg.predict(X_test), y_test)
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
    if evaluation_type == 'holdout':
        return holdout_evaluation(configuration)
    else:
        return cv_evaluation(configuration)


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
        configs.append(_config)
        results.append(_val)

    inc_idx = np.argmin(results)
    return configs[inc_idx], (configs, results)


if __name__ == "__main__":
    inc, details = tune_hyperparamerer_smac()
    configs, results = details
    idx = np.argmin(results)
    print('-'*50)
    print('Results for all configurations evaluated', results)
    print('The best configuration found is', configs[idx])
