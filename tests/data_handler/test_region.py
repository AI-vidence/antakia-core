from unittest import TestCase

import pandas as pd

from antakia_core.data_handler import Region, Rule, RuleSet, RegionSet, ModelRegion, ModelRegionSet
from antakia_core.utils import Variable
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask


class TestRegion(TestCase):

    def setUp(self) -> None:
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

        self.r1_1 = Rule(self.v1,
                         min=2,
                         includes_min=True,
                         max=10,
                         includes_max=False)

        self.r2_1 = Rule(self.v2,
                         min=1.5,
                         includes_min=False)

    def test_init(self):
        """
        Tests that the region is correctly instanciated,
        wether a mask or a rule is given or not

        Tests the implementation of property and setter : color, name

        """

        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert region.num < 0
        pd.testing.assert_series_equal(region.mask,
                                       rule_set.get_matching_mask(self.X))

        # test property and setter : color
        assert region._color is None
        assert not region.validated
        assert not region.auto_cluster
        region.color = 'Blue'
        assert region.color == 'Blue'

        assert region.name == '2.00 ≤ var1 < 10.00'

    def test_to_dict(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert region.to_dict() == {
            'Region': -1,
            'Rules': '2.00 ≤ var1 < 10.00',
            'Average': None,
            'Points': 2,
            '% dataset': '40.0%',
            'Sub-model': None,
            'color': 'cyan'
        }

    def test_numpoint(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert region.num_points() == 2

    def test_dataset_cov(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert region.dataset_cov() == 0.4

    def test_validate(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        region.validate()
        assert region.validated

    def test_stats(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert region.num_points() == 2
        assert region.dataset_cov() == 2 / 5

    def test_update_rule_set(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)

        rule_set2 = RuleSet([self.r2_1])
        region.update_rule_set(rule_set2)
        assert self.r2_1 in region.rules.rules.values()
        assert len(region.rules) == 1

        pd.testing.assert_series_equal(region.mask,
                                       rule_set2.get_matching_mask(self.X))

    def test_update_mask(self):
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        mask = dummy_mask(self.X)
        region.update_mask(mask)
        assert region.mask.equals(mask)
        assert len(region.rules) == 0


    def test_get_color_series(self):
        pass

class TestModelRegion(TestCase):

    def setUp(self):
        X, y = generate_corner_dataset(10)
        X_test, y_test = generate_corner_dataset(10)

        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.X_test = pd.DataFrame(X_test)
        self.y_test = pd.DataFrame(y_test)
        self.ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                                  self.costumer_model)
        # self.test_mask = None
        # self.costumer_model =

    def test_init(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model)

    def test_to_dict(self):
        assert ModReg.to_dict().equals()


