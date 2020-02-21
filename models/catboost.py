import numpy as np
from catboost import CatBoostRegressor
from utils.smape import smape


class catboostRegressor():
    def __init__(self, n_estimators, learning_rate, max_depth,
                 subsample, colsample_bylevel, reg_lambda, loss_function):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.colsample_bylevel = colsample_bylevel
        self.loss_function = loss_function

        self.thread_count = -1
        self.estimator = None

    def fit(self, X, y, metric=smape):
        self.estimator = CatBoostRegressor(max_depth=self.max_depth,
                                           learning_rate=self.learning_rate,
                                           n_estimators=self.n_estimators,
                                           objective='regression',
                                           subsample=self.subsample,
                                           colsample_bylevel=self.colsample_bylevel,
                                           reg_lambda=self.reg_lambda,
                                           thread_count=self.thread_count,
                                           loss_function=self.loss_function)
        self.estimator.fit(X, y, eval_metric=metric)
        return self

    def predict(self, X):
        return self.estimator.predict(X)
