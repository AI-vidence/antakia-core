from unittest import TestCase
import pandas as pd

from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.utils.variable import Variable, DataVariables
from antakia_core.data_handler.rules import RuleSet


class TestSkopeRule(TestCase):

    def setUp(self):
        self.X = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ],
            columns=['var1', 'var2'])

        self.var1 = Variable(0, 'var1', 'float')
        self.var2 = Variable(0, 'var2', 'float')
        self.variables = DataVariables([self.var1, self.var2])

    def test_skope_rules_max_rule(self):
        # test for a max rule
        mask1 = pd.Series([True, True, False, False, False])
        sk_rule = skope_rules(mask1, self.X, self.variables)
        assert isinstance(sk_rule[0], RuleSet)
        assert isinstance(sk_rule[1], dict)

    def test_skope_rules_min_rule(self):
        mask2 = pd.Series([False, False, False, True, True])
        sk_rule1 = skope_rules(mask2, self.X, self.variables)
        assert isinstance(sk_rule1[0], RuleSet)
        assert isinstance(sk_rule1[1], dict)

    def test_skope_rules_interval_rule(self):
        mask2 = pd.Series([False, False, True, True, False])
        sk_rule2 = skope_rules(mask2, self.X, self.variables)
        assert isinstance(sk_rule2[0], RuleSet)
        assert isinstance(sk_rule2[1], dict)

    def test_skope_rules_variable_None(self):
        mask1 = pd.Series([True, True, False, False, False])
        sk_rule3 = skope_rules(mask1, self.X)
        assert isinstance(sk_rule3[0], RuleSet)
        assert isinstance(sk_rule3[1], dict)

    def test_skope_rules_impossible_rule(self):
        # test for impossible rule, check if rs and dict are empty
        mask3 = pd.Series([True, True, False, True, True])
        sk_rule = skope_rules(mask3, self.X, self.variables)
        assert len(sk_rule[0]) == 0
        assert sk_rule[1] == {}
