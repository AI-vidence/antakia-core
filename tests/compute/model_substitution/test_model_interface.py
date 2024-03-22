import unittest
from typing import List
from unittest import TestCase

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

from antakia_core.compute.model_subtitution.classification_models import *
from antakia_core.compute.model_subtitution.model_interface import InterpretableModels
from antakia_core.compute.model_subtitution.regression_models import *
import re

from antakia_core.utils.utils import ProblemCategory
from tests.dummy_datasets import generate_corner_dataset


class TestInterpretableModels(TestCase):
    def setUp(self):
        X, y = generate_corner_dataset(10, random_seed=1234)
        X_test, y_test = generate_corner_dataset(10, random_seed=4321)

        self.X = pd.DataFrame(X, columns=['var1', 'var2'])
        self.y = pd.Series(y)
        self.X_test = pd.DataFrame(X_test)
        self.y_test = pd.Series(y_test)

    def test_init(self):
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
        # test with a scoring method wich has a compute method
        int_mod = InterpretableModels(mean_squared_error)
        int_mod._init_scores(LinearRegression, ProblemCategory.regression, X_test=self.X_test, y_test=self.y_test)
        # assert int_mod.scores == {}

        # test with a regression model
        # int_mod = InterpretableModels('MSE')
        # int_mod._init_scores()
        # assert int_mod.scores == {}
        #
        # test with not a
        # int_mod = InterpretableModels('MSE')
        # int_mod._init_scores()
        # assert int_mod.scores == {}

    def test_train_models(self):  # not ok
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        int_mod._train_models(self.X, self.y, self.X_test, self.y_test)

    def test_compute_score_type(self):  # not ok
        pass

    def test_get_model_performance(self):  # not ok
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        int_mod.get_models_performance(LinearRegression, self.X, self.y, self.X_test, self.y_test)

    def test_select_model(self):
        int_mod = InterpretableModels('MSE')
        int_mod.select_model(LinearRegression())
        assert int_mod.selected_model == LinearRegression

    def test_selected_model_str(self):  # not ok
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        int_mod.select_model('Linear Regression')
        assert int_mod.selected_model_str()

    def test_reset(self):  # not ok
        int_mod = InterpretableModels('MSE')
        int_mod._init_models(ProblemCategory.regression)
        # init scores
        # init perfs
        int_mod.select_model(LinearRegression)
        assert not int_mod.models == {}
        assert not int_mod.selected_model is None
        int_mod.reset()
        assert int_mod.models == {}
        assert int_mod.selected_model is None
