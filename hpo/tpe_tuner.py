import os
import gc
import sys
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

sys.path.append(os.getcwd())
from utils.dataset_loader import load_raw_task_data
from utils.smape import smape

# ---CHANGE SETTINGS HERE---------------
regressor_id = 'liblinear_svr'
trial_num = 15
# evaluation_type = ['holdout', 'cv']
evaluation_type = 'holdout'


# -----------------------------


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
X, y, _ = load_raw_task_data()


def holdout_evaluation(configuration):
    print(configuration)
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
    print(idx)
    print('Results for all configurations evaluated', results)
    print('The best configuration found is', configs[idx])