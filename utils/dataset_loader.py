import pandas as pd

train_data = pd.read_csv('data/candidate_train.csv')
train_answer = pd.read_csv('data/train_answer.csv')
test_data = pd.read_csv('data/candidate_val.csv')

print('complete dataset loading...')
train_data = train_data.merge(train_answer, on='id', how='left')
train_raw_data = train_data.iloc[:, 0:3177].values
test_raw_data = test_data.iloc[:, 0:3177].values
print(train_raw_data.shape, test_raw_data.shape)


def load_raw_task_data(task_id=3):
    label = train_data['p%d' % task_id].values
    print(train_raw_data.shape, label.shape)
    print(test_raw_data.shape)
    return train_raw_data, label, test_raw_data
