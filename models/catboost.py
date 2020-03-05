import numpy as np
from catboost import CatBoostRegressor as CBR
from utils.smape import smape


class CatBoostRegressor():
    def __init__(self, n_estimators, learning_rate, max_depth,
                 subsample, reg_lambda, loss_function, random_state, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.loss_function = loss_function

        self.thread_count = -1
        self.random_state = random_state
        self.other_configs = kwargs
        self.estimator = None

    def fit(self, X, y, metric=smape):
        self.estimator = CBR(max_depth=self.max_depth,
                             learning_rate=self.learning_rate,
                             n_estimators=self.n_estimators,
                             objective='regression',
                             subsample=self.subsample,
                             reg_lambda=self.reg_lambda,
                             thread_count=self.thread_count,
                             loss_function=self.loss_function,
                             random_state=self.random_state, **self.other_configs)
        self.estimator.fit(X, y, eval_metric=metric)
        return self

    def predict(self, X):
        return self.estimator.predict(X)
