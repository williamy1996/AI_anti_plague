import numpy as np
from catboost import CatBoostRegressor
from utils.smape import smape


class catboostRegressor():
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_samples,
                 subsample, colsample_bylevel, reg_alpha, reg_lambda, loss_function):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.colsample_bylevel = colsample_bylevel
        self.loss_function = loss_function

        self.n_jobs = -1
        self.estimator = None

    def fit(self, X, y, metric=smape):
        self.estimator = CatBoostRegressor(num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       objective='regression',
                                       min_child_samples=self.min_child_samples,
                                       subsample=self.subsample,
                                       colsample_bylevel=self.colsample_bylevel,
                                       reg_alpha=self.reg_alpha,
                                       reg_lambda=self.reg_lambda,
                                       n_jobs=self.n_jobs
                                       loss_function=self.loss_function)
        self.estimator.fit(X, y, eval_metric=metric)
        return self

    def predict(self, X):
        return self.estimator.predict(X)