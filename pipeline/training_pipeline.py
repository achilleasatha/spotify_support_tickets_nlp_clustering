import pandas as pd
import numpy as np
import os
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import pickle


class TrainingPipeline:
    def __init__(self, model, data_path, param_grid=None, data=None, random_state=1337):
        self.X, self.y = [None, None]
        self.model = model
        self.param_grid = param_grid
        if data is None:
            self.data = pd.read_csv(data_path + r'\modelling_data.csv')
        else:
            self.data = data
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.3,
                                                                                random_state=self.random_state)
        # rebalance?  # todo
        self.pipeline = Pipeline([
                                  ('preprocess', 'preprocessing'),
                                  ('model', model(random_state=random_state))
                                  ])  # todo
        print('Training...')
        self.pipeline.fit(self.X_train, self.y_train)
        self.train_predictions,  self.test_predictions = self.model.predict(self.X_train), self.model.predict(self.X_test)
        if param_grid is not None:
            self.crossval_gridsearch = self.hyperparam_tuning(self.pipeline, self.X_train, self.y_train, self.param_grid)
        # cross_validation
        # results report
        self.pickle_model()

    @staticmethod
    def hyperparam_tuning(model, X_train, y_train, param_grid, cv=10, scoring='precision_weighted'):
        print('Hyperparameter tuning with gridsearch...')
        crossval_gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
        crossval_gridsearch.fit(X_train, y_train)
        print('-' * 10)
        print('Best Parameters: ', crossval_gridsearch.best_params_)
        print('Mean of Cross Validated Scores: ', crossval_gridsearch.best_score_)
        return crossval_gridsearch

    def pickle_model(self):
        pickle.dump(self.model, open(os.getcwd() + r'\model.pkl', 'wb'))