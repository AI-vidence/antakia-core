from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from antakia_core.utils.utils import ProblemCategory

from antakia_core.explanation.explanations import compute_explanations
from sklearn.tree import DecisionTreeRegressor

from tests.dummy_datasets import load_dataset
from tests.utils_fct import DummyProgress, DummyModel


class TestComputeExplanation(TestCase):

    def setUp(self):
        self.X, self.y = load_dataset('Corner', 100, random_seed=42, columns=4)
        self.model_DT = DecisionTreeRegressor().fit(self.X, self.y)
        self.model_any = DummyModel().fit(self.X, self.y)
        self.X = self.X.sample(100)  # randomize index order

    def test_compute_explanations_DT(self):
        """
        run compute explanation with all explanation methods and check output format
        Returns
        -------

        """
        for i in range(4):
            if i in (1, 2):
                X_exp = compute_explanations(self.X, self.model_DT, i,
                                             ProblemCategory.regression,
                                             DummyProgress())
                pd.testing.assert_index_equal(X_exp.index, self.X.index)
                pd.testing.assert_index_equal(X_exp.columns, self.X.columns)
            else:
                with pytest.raises(ValueError):
                    compute_explanations(self.X, self.model_DT,
                                         ProblemCategory.regression, i,
                                         DummyProgress())

    def test_compute_explanations_dummy(self):
        """
        run compute explanation with all explanation methods and check output format
        Returns
        -------

        """
        for i in range(4):
            if i in (1, 2):
                X_exp = compute_explanations(self.X, self.model_any, i,
                                             ProblemCategory.regression,
                                             DummyProgress())
                pd.testing.assert_index_equal(X_exp.index, self.X.index)
                pd.testing.assert_index_equal(X_exp.columns, self.X.columns)
            else:
                with pytest.raises(ValueError):
                    compute_explanations(self.X, self.model_any,
                                         ProblemCategory.regression, i,
                                         DummyProgress())
