import unittest

import pandas as pd

from antakia_core.data_handler.rules import Rule, RuleSet
from antakia_core.data_handler.region import Region
from antakia_core.data_handler.region_set import RegionSet
from antakia_core.utils.variable import Variable


class TestRegion(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ], columns=['var1', 'var2'])
        self.v1 = Variable(0, 'var1', 'float')
        self.v2 = Variable(0, 'var2', 'float')

        self.r1_1 = Rule(self.v1, max=10, includes_max=False)
        self.r1_2 = Rule(self.v1, min=2, includes_max=True)

        self.r2_1 = Rule(self.v2, min=1.5, includes_min=False)

    def test_init(self):
        rule_set = RuleSet([self.r1_1, self.r1_2])
        region = Region(self.X, rule_set)
        assert region.num < 0
        assert pd.testing.assert_series_equal(region.mask, rule_set.get_matching_mask(self.X))
        assert region.color is None
        assert region.validated is None
        assert region.auto_cluster is None

    def test_color(self):
        rule_set = RuleSet([self.r1_1, self.r1_2])
        region = Region(self.X, rule_set)
        assert region.color is not None
        assert region._color is None
        region.color = 'red'
        assert region.color == 'red'
        assert region._color == 'red'

    def test_to_dict(self):
        rule_set = RuleSet([self.r1_1, self.r1_2])
        region = Region(self.X, rule_set)
        assert isinstance(region.to_dict(), dict)

    def test_stats(self):
        rule_set = RuleSet([self.r1_1, self.r1_2])
        region = Region(self.X, rule_set)
        assert region.num_points() == 2
        assert region.dataset_cov() == 2 / 5

    def test_update_rule_set(self):
        rule_set = RuleSet([self.r1_1, self.r1_2])
        region = Region(self.X, rule_set)

        rule_set2 = RuleSet([self.r2_1])
        region.update_rule_set(rule_set2)
        assert self.r2_1 in region.rules
        assert len(region.rules) == 0

        assert pd.testing.assert_series_equal(region.mask, rule_set2.get_matching_mask(self.X))


class TestModelRegion(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ], columns=['var1', 'var2'])
        self.v1 = Variable(0, 'var1', 'float')
        self.v2 = Variable(0, 'var2', 'float')

        self.r1_1 = Rule(self.v1, max=10, includes_max=False)
        self.r1_2 = Rule(self.v1, min=2, includes_max=True)

        self.r2_1 = Rule(self.v2, min=1.5, includes_min=False)


def test_regions():
    data = pd.DataFrame([
        [1, 2],
        [2, 1],
        [4, 2],
        [10, 1],
        [20, 2],
    ], columns=['var1', 'var2'])
    rs = RegionSet(data)
    assert len(rs) == 0
    var = Variable(0, 'var1', 'float')
    var2 = Variable(0, 'var2', 'float')
    rule1 = Rule(var, max=10, includes_max=False)
    rule2 = Rule(var2, min=2, includes_min=True)

    region = rs.add_region(rules=RuleSet([rule1, rule2]))
    assert rs.get(1) == region
    assert len(rs) == 1
    assert rs.get_max_num() == 1
    assert rs.get_new_num() == 2

    color = rs.get_color_serie()
    assert (color == pd.Series(['red', 'grey', 'red', 'grey', 'grey'])).all()

    rs.clear_unvalidated()
    rs.add_region(RuleSet([rule1, rule2]), color='blue')
    assert rs.get_max_num() == 1
    assert rs.get_new_num() == 2
    assert rs.get(1).color == 'blue'

    color = rs.get_color_serie()
    assert (color == pd.Series(['blue', 'grey', 'blue', 'grey', 'grey'])).all()

    var2 = Variable(0, 'var2', 'float')
    rule3 = Rule(var2, min=1.5, includes_min=False)
    rs.add_region(RuleSet([rule3]), color='blue')
    rs.stats()

    r = Region(data, RuleSet([rule3]))
    r.num = -1

    rs.add(r)
    assert r.num == 3
