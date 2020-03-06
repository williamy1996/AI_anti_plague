import numpy as np
from catboost import CatBoostRegressor
from utils.smape import smape


class CBRegressor():
    def __init__(self, n_estimators, learning_rate, max_depth,
                 subsample, reg_lambda, loss_function, random_state, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.loss_function = loss_function
        self.colsample_bylevel = kwargs.get('colsample_bylevel', None)
        self.min_child_samples = kwargs.get('min_child_samples', 1)

        self.thread_count = -1
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.estimator = CatBoostRegressor(max_depth=self.max_depth,
                                           learning_rate=self.learning_rate,
                                           n_estimators=self.n_estimators,
                                           objective='regression',
                                           subsample=self.subsample,
                                           reg_lambda=self.reg_lambda,
                                           thread_count=self.thread_count,
                                           loss_function=self.loss_function,
                                           colsample_bylevel=self.colsample_bylevel,
                                           random_state=self.random_state)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)
