from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    precision_score, recall_score

from antakia_core.compute.model_subtitution.classification_models import *
from antakia_core.compute.model_subtitution.model_interface import InterpretableModels
from antakia_core.compute.model_subtitution.regression_models import *

from antakia_core.utils.utils import ProblemCategory


class TestInterpretableModels(TestCase):
    def setUp(self):
        X = np.random.randn(500, 4)
        y = np.sum(X, axis=1)
        self.X_train = pd.DataFrame(X[:250], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_train = pd.Series(y[:250])

        self.X_test = pd.DataFrame(X[250:], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_test = pd.Series(y[250:])

    def test_init(self):  # not ok
        pass

    def test_get_available_models(self):
        int_mod = InterpretableModels('MSE')
        assert LinearRegression in int_mod._get_available_models(ProblemCategory.regression)
        assert AvgClassificationBaselineModel in int_mod._get_available_models(ProblemCategory.auto)

    def test_init_models(self):
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        assert isinstance(int_mod.models, dict)
        assert len(int_mod.models) == 7

    def test_init_scores(self):  # not ok
        int_mod = InterpretableModels(mean_squared_error)
        int_mod._init_models(ProblemCategory.regression)

        # test with a regression model
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)

        # test with a classification model
        model = int_mod.models['Decision Tree']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.classification, X_test=self.X_test, y_test=self.y_test)

    def test_train_models(self):
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)
        int_mod._train_models(self.X_train, self.y_train, self.X_test, self.y_test)

    def test_compute_score_type(self):
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._compute_score_type(model, self.X_train, self.y_train)
        assert int_mod.score_type == 'minimize'

    def test_get_model_performance(self):
        # inits
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)

        # test standard case
        perfs = int_mod.get_models_performance(model, self.X_train, self.y_train, self.X_test, self.y_test)
        assert not perfs.equals(pd.DataFrame())

        # test with too small training dataset
        perfs = int_mod.get_models_performance(model, self.X_train[:40], self.y_train[:40], self.X_test, self.y_test)
        assert perfs.equals(pd.DataFrame())

        # test with no test set provided
        perfs = int_mod.get_models_performance(model, self.X_train, self.y_train, X_test=None, y_test=None)
        assert not perfs.equals(pd.DataFrame())

    def test_select_model(self):
        int_mod = InterpretableModels('MSE')
        int_mod.select_model('Linear Regression')
        assert int_mod.selected_model == 'Linear Regression'

    def test_selected_model_str(self):
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)
        int_mod.get_models_performance(model, self.X_train, self.y_train, self.X_test, self.y_test)

        int_mod.select_model('Linear Regression')
        assert int_mod.selected_model_str() == 'LR - MSE:0.00 (0.00)'

    def test_reset(self):
        # init
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        model = int_mod.models['Linear Regression']
        model.fit(self.X_train, self.y_train)
        int_mod._init_scores(model, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)
        int_mod.get_models_performance(model, self.X_train, self.y_train, self.X_test, self.y_test)
        int_mod.select_model('Linear Regression')
        # checks that attributes are initialized
        assert not int_mod.models == {}
        assert not int_mod.scores == {}
        assert not int_mod.perfs.equals(pd.DataFrame())
        assert not int_mod.selected_model is None
        int_mod.reset()
        # checks that attributes are correctly reset
        assert int_mod.models == {}
        assert int_mod.scores == {}
        assert int_mod.perfs.equals(pd.DataFrame())
        assert int_mod.selected_model is None
