import pandas as pd


def load_raw_task_data(task_id=3):
    train_data = pd.read_csv('data/candidate_train.csv')
    train_answer = pd.read_csv('data/train_answer.csv')

    train_data = train_data.merge(train_answer, on='id', how='left')
    print(train_data.head())

    train_raw_data = train_data.iloc[:, 0:3177].values
    label = train_data['p%d' % task_id].values
    print(train_raw_data.shape, label.shape)
    return train_raw_data, label
