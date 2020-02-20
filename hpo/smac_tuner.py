import os
import sys
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import train_test_split
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

sys.path.append(os.getcwd())
from utils.dataset_loader import load_raw_task_data
from utils.smape import smape


reg_id = 'liblinear_svr'
trial_num = 10


def create_hyperspace(reg_id):
    if reg_id == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        cs = KNearestNeighborsRegressor.get_hyperparameter_search_space()
    elif reg_id == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        cs = LibLinear_SVR.get_hyperparameter_search_space()
    elif reg_id == 'your_reg_id':
        cs = ConfigurationSpace()
        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])
    else:
        raise ValueError('Undefined regressor identifier: %s!' % reg_id)
    return cs


def get_reg(reg_id, config):
    if reg_id == 'knn':
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        reg = KNearestNeighborsRegressor(**config.get_dictionary())
    elif reg_id == 'liblinear_svr':
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        reg = LibLinear_SVR(**config.get_dictionary())
    elif reg_id == 'your_reg_id':
        # Based on the hyperparameter configuration `config`, construct the regressor.
        pass
    else:
        raise ValueError('Undefined regressor identifier: %s!' % reg_id)
    return reg


# Load data.
X, y = load_raw_task_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)


def obj_func_example(configuration):
    reg = get_reg(reg_id, configuration)
    # Fit this regressor on the train data.
    print('Starting to fit a regression model - %s' % reg_id)
    reg.fit(X_train, y_train)
    return smape(reg.predict(X_test), y_test)


def tune_hyperparamerer_smac():
    configs, results = list(), list()
    config_space = create_hyperspace(reg_id)
    scenario_dict = {
        'abort_on_first_run_crash': False,
        "run_obj": "quality",
        "cs": config_space,
        "deterministic": "true",
        "runcount-limit": trial_num
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
    print(inc)
    print('-'*50)
    configs, results = details
    idx = np.argmin(results)
    print('All results', results)
    print('INC', configs[idx], results[idx])
