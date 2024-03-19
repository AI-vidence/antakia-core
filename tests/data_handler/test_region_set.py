import unittest

import pandas as pd

from antakia_core.data_handler import Rule, RegionSet, RuleSet, ModelRegionSet
from antakia_core.utils import Variable, ProblemCategory
from tests.utils_fct import DummyModel


class TestRegionSet(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ],
                              columns=['var1', 'var2'])
        self.v1 = Variable(0, 'var1', 'float')
        self.v2 = Variable(0, 'var2', 'float')

        self.r1_1 = Rule(self.v1, max=10, includes_max=False)
        self.r1_2 = Rule(self.v1, min=2, includes_max=True)
        self.rs1 = RuleSet([self.r1_1, self.r1_2])
        self.r2_1 = Rule(self.v2, min=1.5, includes_min=False)

        self.region_set = RegionSet(self.X)

    def test_left_out(self):
        pass


class TestModelRegionSet(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ],
                              columns=['var1', 'var2'])
        self.y = pd.Series([1, 2, 1, 2, 1])
        self.X_test = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ],
                                   columns=['var1', 'var2'],
                                   index=range(5, 10))
        self.y_test = pd.Series([1, 2, 1, 2, 1], index=range(5, 10))
        self.model = DummyModel()
        self.v1 = Variable(0, 'var1', 'float')
        self.v2 = Variable(0, 'var2', 'float')

        self.r1_1 = Rule(self.v1, max=10, includes_max=False)
        self.r1_2 = Rule(self.v1, min=2, includes_max=True)
        self.rs1 = RuleSet([self.r1_1, self.r1_2])
        self.r2_1 = Rule(self.v2, min=1.5, includes_min=False)

        self.region_set = ModelRegionSet(self.X, self.y, self.X_test,
                                         self.y_test, self.model,
                                         lambda *args: 1)
        self.problem_category = ProblemCategory.regression

    def test_left_out(self):
        self.region_set.left_out_region.train_substitution_models(
            task_type=self.problem_category)
