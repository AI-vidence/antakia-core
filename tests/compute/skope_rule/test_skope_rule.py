from unittest import TestCase
import pandas as pd
import pytest

from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.utils.variable import Variable, DataVariables
from antakia_core.data_handler.rules import Rule, RuleSet
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask


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

    def test_skope_rules__test_sk_rules_to_rule_set(self):
        # test for a max rule
        mask1 = pd.Series([True, True, False, False, False])
        sk_rule = skope_rules(mask1, self.X, self.variables)
        assert isinstance(sk_rule[0], RuleSet)
        assert isinstance(sk_rule[1], dict)

        # test for a min rule
        mask2 = pd.Series([False, False, False, True, True])
        sk_rule1 = skope_rules(mask2, self.X, self.variables)
        assert isinstance(sk_rule1[0], RuleSet)
        assert isinstance(sk_rule1[1], dict)

        # test for interval rule
        mask2 = pd.Series([False, False, True, True,False ])
        sk_rule2 = skope_rules(mask2, self.X, self.variables)
        assert isinstance(sk_rule2[0], RuleSet)
        assert isinstance(sk_rule2[1], dict)


        # test when variables is None
        sk_rule3 = skope_rules(mask1, self.X)
        assert isinstance(sk_rule3[0], RuleSet)
        assert isinstance(sk_rule3[1], dict)

        # test for impossible rule, check if rs and dict are empty
        mask3 = pd.Series([True, True, False, True, True])
        sk_rule = skope_rules(mask3, self.X, self.variables)
        assert len(sk_rule[0]) == 0
        assert sk_rule[1] == {}

        #test for unrecognized rule
        df = pd.DataFrame([
            [1, 7],
            [2, 3],
            [4, 5],
            [10, 4],
            [20, 2],
        ],
            columns=['var1', 'var2'])

        mask4 = pd.Series([False, False, True, True, False])
        sk_rule4 = skope_rules(mask4, df, self.variables)

        # with pytest.raises(ValueError):
        #     sk_rule = skope_rules(mask4, self.X, self.variables)
        #
        # test for falsy rule
        # mask = pd.Series([False, False, False, False, False])
        # sk_rule2 = skope_rules(mask, self.X, self.variables)

        # test for truthy rule
        # mask = pd.Series([True, True, True,True,True])
        # sk_rule3 = skope_rules(mask, self.X, self.variables)
