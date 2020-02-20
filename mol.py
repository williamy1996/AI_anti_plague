from sklearn.model_selection import train_test_split
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import roc_curve,classification_report
from  sys import argv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.decomposition import PCA

def zero_select(X_train,num):
	X_bool = (X_train!=0)
	fs = np.sum(X_bool,axis=0)<num
	sub = []
	for i in range(3177):
		if(fs[i]==False):
			sub.append(i)
	return sub
#选取非零个数大于num的特征

def rmae(y_true, y_pred):
    print(y_true)
    down = (abs(y_true)+abs(y_pred))/2+1e-6
    up = abs(y_true-y_pred)
    v = up/down
    return np.average(v)
#计算目标值

def write_csv(id_,p1,p2,p3,p4,p5,p6,csv_name):
	id_ = pd.Series(id_,name='id')
	p1 = pd.Series(p1,name='p1')
	p2 = pd.Series(p2,name='p2')
	p3 = pd.Series(p3,name='p3')
	p4 = pd.Series(p4,name='p4')
	p5 = pd.Series(p5,name='p5')
	p6 = pd.Series(p6,name='p6')
	con = pd.concat([p1,p2,p3],axis=1)
	save.to_csv(csv_name)
#写csv文件

data = np.array(pd.read_csv('moldata/molecule_open_data/candidate_train.csv',encoding='gbk'))
m,n = data.shape
ans = np.array(pd.read_csv('moldata/molecule_open_data/train_answer.csv',encoding='gbk'))
print(m,n)
data = np.hstack([data[:,3177:],data[:,0:3177]])
data = data[data[:,0].argsort()]
ans = ans[ans[:,0].argsort()]
X_set = data[:,1:]
print(data)
print(ans)
X_set = np.float32(X_set)
y_set = ans[:,1:]
y_set = np.float32(y_set)
m_y = np.max(abs(y_set),axis=0)
print(m_y)
y_set = y_set/m_y
X_train,X_test,y_train,y_test = train_test_split(X_set,y_set,test_size=0.2)
reg = RandomForestRegressor(n_estimators=1000,max_depth=20, random_state=0)
#xgb1 = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
#xgb1 = XGBClassifier(learning_rate = 0.1,n_estimators=500,max_depth=6,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,eval_metric='auc',seed=27)
reg.fit(X_train,y_train[:,2]>0)