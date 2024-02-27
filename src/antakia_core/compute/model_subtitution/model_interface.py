from typing import List

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

from antakia_core.compute.model_subtitution.classification_models import *
from antakia_core.compute.model_subtitution.regression_models import *
import re

from antakia_core.utils.utils import ProblemCategory


def pretty_model_name(model_name):
    return model_name.replace('_', ' ').title()


def reduce_name(model_name):
    parts = re.split('\W+', model_name)
    name = ''
    for part in parts:
        name += part[0].upper()
        for char in part[1:]:
            if char.isupper():
                name += char
    if len(name) == 1:
        return model_name[:2].capitalize()
    return name


class InterpretableModels:
    available_scores = {
        'MSE': (mean_squared_error, 'minimize'),
        'MAE': (mean_absolute_error, 'minimize'),
        'R2': (r2_score, 'maximize'),
        'ACC': (accuracy_score, 'maximize'),
        'ACCURACY': (accuracy_score, 'maximize'),
        'F1': (f1_score, 'maximize'),
        'precision'.upper(): (precision_score, 'maximize'),
        'recall'.upper(): (recall_score, 'maximize'),
    }
    customer_model_name = pretty_model_name('original_model')

    def __init__(self, custom_score):
        if callable(custom_score):
            self.custom_score_str = custom_score.__name__.upper()
            self.custom_score = custom_score
            self.score_type = 'compute'
        else:
            self.custom_score_str = custom_score.upper()
            self.custom_score = self.available_scores[custom_score.upper()][0]
            self.score_type = self.available_scores[custom_score.upper()][1]

        self.models = {}
        self.scores = {}
        self.perfs = pd.DataFrame()
        self.selected_model = None

    def _get_available_models(self, task_type) -> List[type[MLModel]]:
        if task_type == ProblemCategory.regression:
            return [LinearRegression, LassoRegression, RidgeRegression, GaM,
                    EBM, DecisionTreeRegressor, AvgBaselineModel]
        return [AvgClassificationBaselineModel, DecisionTreeClassifier, LogisticRegression]

    def _init_models(self, task_type):
        for model_class in self._get_available_models(task_type):
            model = model_class()
            if model.name not in self.models:
                self.models[pretty_model_name(model.name)] = model

    def _init_scores(self, customer_model, task_type):
        if self.score_type == 'compute':
            self._compute_score_type(customer_model, X_test, y_test)
        if task_type == ProblemCategory.regression:
            scores_list = ['MSE', 'MAE', 'R2']
        else:
            scores_list = ['ACC', 'F1', 'precision'.upper(), 'recall'.upper(), 'R2']
        self.scores = {
            score: self.available_scores[score] for score in scores_list
        }
        self.scores[self.custom_score_str] = (self.custom_score, self.score_type)

    def _train_models(self, X_train, y_train, X_test, y_test):
        Parallel(n_jobs=1)(
            delayed(model.fit_and_compute_fi)(X_train, y_train, X_test, y_test, self.custom_score, self.score_type) for
            model_name, model in
            self.models.items() if not model.fitted)

    def _compute_score_type(self, customer_model, X: pd.DataFrame, y: pd.Series):
        y_pred = customer_model.predict(X)
        s1 = self.custom_score(y_pred, y)
        s2 = self.custom_score(y.sample(len(y)).values, y.values)
        self.score_type = 'maximize' if s1 > s2 else 'minimize'

    def get_models_performance(
            self,
            customer_model,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame | None,
            y_test: pd.Series | None,
            task_type='regression') -> pd.DataFrame:
        if isinstance(task_type, str):
            task_type = ProblemCategory[task_type]
        if len(X_train) <= 50 or len(X_train.T) >= len(X_train):
            return pd.DataFrame()
        if X_test is None or len(X_test) == 0:
            print('no test set provided, splitting train set')
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        self._init_scores(customer_model, task_type)
        self._init_models(task_type)
        if customer_model is not None:
            self.models[self.customer_model_name] = MLModel(customer_model, self.customer_model_name, True)
            self.models[self.customer_model_name].compute_feature_importances(X_test, y_test, self.custom_score,
                                                                              self.score_type)
        self._train_models(X_train, y_train, X_test, y_test)
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            for score_name, score in self.scores.items():
                self.perfs.loc[model_name, score_name] = score[0](y_test, y_pred)

        self.perfs['delta'] = (self.perfs[self.custom_score_str] - self.perfs.loc[
            self.customer_model_name, self.custom_score_str]) * (2 * (self.score_type == 'minimize') - 1)

        def get_delta_color(delta):
            return 'red' if delta > 0.01 else 'green' if delta < -0.01 else 'orange'

        self.perfs['delta_color'] = self.perfs['delta'].apply(get_delta_color)
        return self.perfs.sort_values('delta', ascending=True)

    def select_model(self, model_name):
        self.selected_model = model_name

    def selected_model_str(self) -> str:
        perf = self.perfs.loc[self.selected_model]
        reduced_name = reduce_name(self.selected_model)
        display_str = f'{reduced_name} - {self.custom_score_str}:{perf[self.custom_score_str]:.2f} ({perf["delta"]:.2f})'
        return display_str


if __name__ == '__main__':
    df = pd.read_csv('../../../../antakia/data/california_housing.csv').set_index('Unnamed: 0')
    df = df.sample(len(df))
    limit = int(2000 / 0.8)
    df = df.iloc[:limit]
    split_row = int(len(df) * 0.8)
    df_train = df[:split_row]
    df_test = df[split_row:]
    X_train = df_train.iloc[:, :8]  # the dataset
    y_train = df_train.iloc[:, 9]  # the target variable
    X_test = df_test.iloc[:, :8]  # the dataset
    y_test = df_test.iloc[:, 9]  # the target variable
    InterpretableModels(mean_squared_error).get_models_performance(
        None, X_train, y_train, X_test, y_test,
        task_type='regression')
