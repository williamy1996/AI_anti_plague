import numpy as np
import pandas as pd


def save_task_result(mth_id, task_id, pred_y):
    np.save('data/%s-pred_results_task-%d.npy' % (mth_id, task_id), pred_y)


def create_submission_file(mth_id):
    pred_test = list()
    for task_id in range(1, 7):
        _pred = np.load('data/%s-pred_results_task-%d.npy' % (mth_id, task_id))
        pred_test.append(_pred)
    pred_test = np.array(pred_test).transpose()
    from utils.dataset_loader import test_data
    pred_df = pd.DataFrame(pred_test, columns=['p%d' % i for i in range(1, 7)])
    result = pd.concat([test_data[['id']], pred_df], axis=1).reset_index(drop=True)
    result.to_csv('data/%s-result.csv' % mth_id, index=False)
    print(result.head())
