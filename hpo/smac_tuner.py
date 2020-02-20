import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC


def tune_hyperparamerer_smac(config_space, obj_func, trial_num):
    configs, results = list(), list()

    scenario_dict = {
        'abort_on_first_run_crash': False,
        "run_obj": "quality",
        "cs": config_space,
        "deterministic": "true",
        "runcount-limit": trial_num
    }
    optimizer = SMAC(scenario=Scenario(scenario_dict),
                     rng=np.random.RandomState(45),
                     tae_runner=obj_func)

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
