import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC


def smape(y_pred, y_true):
    sample_size = len(y_pred)
    numerator = np.abs(y_pred-y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true))/2
    return np.sum(np.divide(numerator, denominator))/sample_size


def create_hyperspace():
    pass


def obj_func_example(configuration):
    # Prepare the train/valid data.
    x_train, y_train, x_val, y_val = None, None, None, None

    # Build the regressor.
    clf = None
    # Fit this regressor on the train data.

    return smape(clf.predict(x_val), y_val)


def tune_hyperparamerer_smac(trial_num=100):
    configs, results = list(), list()
    config_space = create_hyperspace()
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
    tune_hyperparamerer_smac()
