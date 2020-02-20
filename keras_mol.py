import numpy as np
import random
import xlrd
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import pandas as pd

def rmae(y_true, y_pred):
    print(y_true)
    down = (abs(y_true)+abs(y_pred))/2+1e-6
    up = abs(y_true-y_pred)
    v = up/down
    return K.mean(v)

data = np.array(pd.read_csv('moldata/molecule_open_data/candidate_train.csv',encoding='gbk'))
m,n = data.shape
ans = np.array(pd.read_csv('moldata/molecule_open_data/train_answer.csv',encoding='gbk'))
print(m,n)
data = np.hstack([data[:,3177:],data[:,0:3177]])
data = data[data[:,0].argsort()]
X_train = data[:,1:]
ans = ans[ans[:,0].argsort()]
print(data)
print(ans)
X_train = np.float32(X_train)
y_train = ans[:,1:]
y_train = np.float32(y_train)
m_y = np.max(abs(y_train),axis=0)
print(m_y)
y_train = y_train/m_y
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2)
model = Sequential()
model.add(Dense(32, input_dim=len(X_train[0]), activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_absolute_error',optimizer='rmsprop',metrics=[rmae])
model.fit(X_train, y_train[:,2],epochs=100,batch_size=10000)
score = model.evaluate(X_test,y_test,batch_size=10000)
y_pred2 = model.predict(x_test)