import pandas as pd
import os


def load_raw_task_data(task_id=3):
    train_data = pd.read_csv('data/candidate_train.csv')
    train_answer = pd.read_csv('data/train_answer.csv')
    test_data = pd.read_csv('data/candidate_val.csv')

    print('complete dataset loading...')
    train_data = train_data.merge(train_answer, on='id', how='left')
    train_raw_data = train_data.iloc[:, 0:3177].values
    test_raw_data = test_data.iloc[:, 0:3177].values
    print(train_raw_data.shape, test_raw_data.shape)
    label = train_data['p%d' % task_id].values
    print(train_raw_data.shape, label.shape)
    print(test_raw_data.shape)
    return train_raw_data, label, test_raw_data


def load_holdout_data(data_dir, task_id=3):
    train_data = pd.read_csv(os.path.join(data_dir, 'train_feature.csv'))
    train_data = train_data.drop(columns=["Unnamed: 0", "id"])
    train_data = train_data.values
    valid_data = pd.read_csv(os.path.join(data_dir, 'valid_feature.csv'))
    valid_data = valid_data.drop(columns=["Unnamed: 0", "id"])
    valid_data = valid_data.values
    train_label = pd.read_csv(os.path.join(data_dir, 'train_label.csv'))
    train_label = train_label.drop(columns=["Unnamed: 0"])
    train_label = train_label['p%d' % task_id].values
    valid_label = pd.read_csv(os.path.join(data_dir, 'valid_label.csv'))
    valid_label = valid_label.drop(columns=["Unnamed: 0"])
    valid_label = valid_label['p%d' % task_id].values
    test_data = pd.read_csv(os.path.join(data_dir, 'test_feature.csv'))
    test_id = test_data['id'].values
    test_data = test_data.drop(columns=["Unnamed: 0", "id"])
    print('complete dataset loading...')

    return train_data, valid_data, train_label, valid_label, test_data, test_id


def create_csv_for_fe(task_id=3):
    train_data = pd.read_csv('data/candidate_train.csv')
    train_answer = pd.read_csv('data/train_answer.csv')
    test_data = pd.read_csv('data/candidate_val.csv')

    print('complete dataset loading...')
    train_data = train_data.merge(train_answer, on='id', how='left')
    train_raw_data = train_data.iloc[:, 0:3177]
    test_raw_data = test_data.iloc[:, 0:3177]
    label = train_data['p%d' % task_id]
    result = pd.concat([train_raw_data, label], axis=1).reset_index(drop=True)
    result.to_csv('data/p%s.csv' % task_id, index=False)
    print(result.head())
    test_raw_data.to_csv('data/test_data.csv', index=False)
    print(test_raw_data.head())
