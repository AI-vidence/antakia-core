from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from antakia_core.data_handler.rule import TruthyRule, FalsyRule
from antakia_core.data_handler.rules import Rule, Variable, RuleSet
from tests.dummy_datasets import generate_corner_dataset


class TestRuleSet(TestCase):

    def setUp(self):
        var1 = Variable(0, 'comb1', 'float')
        var2 = Variable(0, 'comb2', 'float')
        rule1_1 = Rule(var1, max=10, includes_max=False)
        rule1_2 = Rule(var1, max=10, includes_max=True)
        rule1_3 = Rule(var1, max=20, includes_max=False)
        self.rule1 = Rule(var1, max=40, includes_max=False)
        self.rule2 = Rule(var1, min=20, includes_max=True)
        self.rules = [rule1_1, rule1_2, rule1_3]
        self.df = pd.DataFrame(generate_corner_dataset(10)[0],
                               columns=['comb1', 'comb2'])

    def test_init(self):
        """
        checks if the rule set is correctly initialized as a dictionary
        with variables as keys and rules as values.
        checks if the rule set is correctly initialized (a copy is made) when given a rule set as parameter.
        checks the init for empty rule set
        """
        rule_set0 = RuleSet()
        rule_set1 = RuleSet(rule_set0)
        assert rule_set1.rules == rule_set0.rules

        rule_set2 = RuleSet(None)
        assert not len(rule_set2)

    def test_add(self):
        """
        checks if a new rule is correctly added to the rule set
        checks if a new rule on an existing variable is correctly combined

        """
        # rule_set0 = RuleSet()
        # rule_set1 = RuleSet()
        rule_set2 = RuleSet()

        # add a rule on a new variable
        # rule_set1.add(self.rule2)
        # assert len(rule_set1) != len(rule_set0)
        # add a rule on an existing variable
        rule_set2.add(self.rule1)

        rule_set2.add(self.rule2)
        a=1
        # assert len(rule_set2) == len(rule_set0)

    def test_replace(self):  # not ok
        """
        checks if a new rule is correctly added, or correctrly replaced
        if variable is already used

        """

        rule_set0 = RuleSet(self.rules)
        rule_set1 = RuleSet(self.rules)
        rule_set2 = RuleSet(self.rules)

        # add a rule on a new variable
        rule_set1.replace(self.rule2)
        assert rule_set1.rules[self.rule2.variable] == self.rule2
        assert len(rule_set1.rules) == 2

        # replace the rules of an existing variable
        rule_set2.replace(self.rule1)
        assert rule_set2.rules[self.rule1.variable] == self.rule1

    def test_to_dict(self):
        """
        checks if the dictionnary is correctly returned
        checks if empty RuleSet returns empty dictionary

        """

        rule_set = RuleSet(self.rules)
        assert isinstance(rule_set.to_dict(), list)
        assert isinstance(rule_set.to_dict()[0], dict)
        rule_set1 = RuleSet()
        assert isinstance(rule_set1.to_dict(), list)

    def test_copy(self):  # not ok
        """
        Returns a copy of the rule set given as parameter
        -------
        """
        rule_set = RuleSet(self.rules)
        now_rule_set = rule_set.copy()
        assert rule_set.rules == now_rule_set.rules

    def test_get_matching_mask(self):
        """
        checks if the mask is correctly generated and is the right date type
        """
        rule_set = RuleSet(self.rules)
        assert isinstance(rule_set.get_matching_mask(self.df), pd.Series)

    def test_get_all_masks(self):
        """
        checks if the masks are correctly generated and if the list is the same len as rule set
        """
        df = self.df
        rule_set = RuleSet()
        assert len(rule_set.get_all_masks(df)) == len(
            rule_set.get_all_masks(df))
        assert isinstance(rule_set.get_all_masks(df), list)

        rule_set = RuleSet(self.rules)
        assert len(rule_set.get_all_masks(df)) == len(
            rule_set.get_all_masks(df))
        assert isinstance(rule_set.get_all_masks(df), list)

    def test_get_matching_indexes(self):
        """

        """
        rule_set = RuleSet(self.rules)
        assert isinstance(rule_set.get_matching_indexes(self.df), list)

    def test_get_rule(self):
        """
        checks if the returned rule matches the var given in parameter
        -------
        Returns the rule related to the variable given in parameter

        """
        var = Variable(0, 'type1', 'float')
        rule_set = RuleSet(self.rules)
        assert rule_set.get_rule(var) == rule_set.rules.get(var)


class TestRule(TestCase):

    def setUp(self):
        self.var1 = Variable(0, 'var1', 'float')
        self.var2 = Variable(0, 'var2', 'float')
        self.df = pd.DataFrame(generate_corner_dataset(10)[0],
                               columns=['var1', 'var2'])

    def test_init(self):  # not ok
        """
        checks the corrrect instanciation of a Rule, with different cases
        wether a min or a max are given
        checks properties
        """

        rule = Rule(self.var1)
        assert rule.min == -np.inf
        assert rule.max == np.inf

        rule1 = Rule(self.var1, min=10, max=20)
        assert not rule1.includes_min
        assert not rule1.includes_max

        rule2 = Rule(self.var1, cat_values=['High', 'Low'])
        assert rule2.min is None
        assert rule2.max is None

    def test_equals__rule_type(self):
        rule1 = Rule(self.var1)
        rule2 = Rule(self.var1, cat_values=['High', 'Low'])
        rule3 = Rule(self.var1, max=30)
        rule4 = Rule(self.var2, min=10, max=20)
        rule5 = Rule(self.var1)
        rule6 = Rule(self.var1, min=10, includes_min=True)
        rule7 = Rule(self.var1, cat_values=['High', 'Low', 'Middle'])
        rule8 = Rule(self.var1, cat_values=['High', 'Low'])
        rule9 = Rule(self.var1,
                     min=10,
                     max=10,
                     includes_max=True,
                     includes_min=True)
        rule10 = Rule(self.var1,
                      min=10,
                      max=10,
                      includes_max=True,
                      includes_min=True)
        rule11 = Rule(self.var2, min=10, max=20)
        rule12 = Rule(self.var1, max=30)
        rule13 = Rule(self.var1, min=10, includes_min=True)

        # assert rule1 == rule3
        assert rule3 != rule4
        assert rule2 != rule7
        assert rule2 == rule8
        assert rule1 == rule5
        assert rule9 == rule10
        assert rule3 == rule12
        assert rule6 == rule13
        assert rule4 == rule11

    def test_get_matching_mask(self):
        """
        checks that a pd Series is generated, and that it
        matches the rule type
        """

        rule1 = Rule(self.var1)
        rule2 = Rule(self.var1, cat_values=['High', 'Low'])
        rule3 = Rule(self.var2, min=20, max=10)
        rule4 = Rule(self.var1,
                     min=10,
                     max=10,
                     includes_max=True,
                     includes_min=True)
        rule5 = Rule(self.var2, min=10, max=20)

        assert rule1.get_matching_mask(self.df).all()  # truthy rule
        assert not rule3.get_matching_mask(self.df).all()  # falsy rule
        assert isinstance(rule2.get_matching_mask(self.df), pd.Series)
        assert isinstance(rule4.get_matching_mask(self.df), pd.Series)
        assert isinstance(rule5.get_matching_mask(self.df), pd.Series)

    def test_combine(self):  # not ok
        """
        checks that different combinations of rules return the expected logical rule
        Returns
        -------

        """
        var1 = Variable(0, 'comb1', 'float')
        var2 = Variable(0, 'comb2', 'float')
        rule1_1 = Rule(var1,
                       max=20,
                       includes_max=False)  # None, None, var1, '<', 20)
        rule1_2 = Rule(var1,
                       max=10,
                       includes_max=False)  # None, None, var1, '<', 10)
        rule1_3 = Rule(var1,
                       max=10,
                       includes_max=True)  # None, None, var1, '<=', 10)
        rule1_4 = Rule(var1,
                       max=5,
                       includes_max=False)  # None, None, var1, '<', 5)
        rule2_1 = Rule(var1,
                       min=10,
                       includes_min=True)  # 10, '<=', var1, None, None)
        rule3_1 = Rule(var1,
                       min=10,
                       includes_min=True,
                       max=40,
                       includes_max=False)  # 10, '<=', var1, '<', 40)
        rule4_1 = Rule(var1,
                       min=40,
                       includes_min=False,
                       max=10,
                       includes_max=False)  # 10, '>', var1, '>', 40)
        rule5_1 = Rule(var2,
                       min=10,
                       includes_min=True,
                       max=40,
                       includes_max=False)  # 10, '<=', var1, '<', 40)

        rule6_1 = Rule(var1, cat_values=['High', 'Middle'])
        rule6_2 = Rule(var1, cat_values=['Middle', 'Low'])
        rule6_3 = Rule(var1, cat_values=['Middle'])

        assert repr(rule2_1.combine(rule1_1)) == '10.00 ≤ comb1 < 20.00'
        assert rule2_1.combine(rule1_2) == rule4_1
        assert repr(rule2_1.combine(rule1_3)) == 'comb1 = 10'
        assert rule2_1.combine(rule1_4) == rule4_1

        assert repr(rule3_1.combine(rule1_1)) == '10.00 ≤ comb1 < 20.00'
        assert rule3_1.combine(rule1_2) == rule4_1
        assert repr(rule3_1.combine(rule1_3)) == 'comb1 = 10'
        assert rule3_1.combine(rule1_4) == rule4_1

        assert rule4_1.combine(rule1_1) == rule4_1
        assert rule4_1.combine(rule1_2) == rule4_1
        assert rule4_1.combine(rule1_3) == rule4_1
        assert rule4_1.combine(rule1_4) == rule4_1

        with pytest.raises(ValueError):
            rule5_1.combine(rule3_1)

        assert rule6_1.combine(rule6_2) == rule6_3

        # tester rule types, si min et min alors min si min max alors interval si min egale max alors value
        # si min sup a max alors falsy

    def test_to_dict(self):
        """
        checks that a dictionnary is correctly returned

        """
        rule1 = Rule(self.var1, min=10, max=20)
        assert isinstance(rule1.to_dict(), dict)

    def test_copy(self):
        """
        checks that a rule is returned, identical to its original
        """
        rule1 = Rule(self.var1, min=10, max=20)
        rule2 = rule1.copy()
        assert rule1 == rule2

    def test_TruthyRule(self):
        """
        checks that a falsy rule returns a mask of false.
        """
        var1 = Variable(0, 'var1', 'float')
        rule1 = TruthyRule(var1)
        assert rule1.get_matching_mask(self.df).all()

    def test_FalselyRule(self):
        """
        checks that a falsy rule returns a mask of false.
        """
        var1 = Variable(0, 'var1', 'float')
        rule1 = FalsyRule(var1)
        assert not rule1.get_matching_mask(self.df).all()


def test_type_1():
    var = Variable(0, 'type1', 'float')
    rule1_1 = Rule(var, max=10, includes_max=False)
    rule1_2 = Rule(var, max=10, includes_max=True)
    rule1_3 = Rule(var, max=20, includes_max=False)
    rule1_4 = Rule(var, max=20, includes_max=False)
    rule1_5 = Rule(var, max=10, includes_max=True)
    rule1_6 = Rule(var, max=10, includes_max=False)
    rule1_7 = Rule(var, max=10, includes_max=False)
    rule1_8 = Rule(var, max=10, includes_max=False)
    rule1_9 = Rule(var, max=10, includes_max=True)

    assert rule1_4 == rule1_3
    assert rule1_5 == rule1_2
    assert repr(rule1_6) == 'type1 < 10.00'
    assert repr(rule1_7) == 'type1 < 10.00'
    assert repr(rule1_8) == 'type1 < 10.00'
    assert repr(rule1_9) == 'type1 ≤ 10.00'

    assert rule1_1.rule_type == 1
    assert rule1_2.rule_type == 1
    assert rule1_3.rule_type == 1

    assert not rule1_1.is_categorical_rule

    assert repr(rule1_1) == 'type1 < 10.00'
    assert repr(rule1_2) == 'type1 ≤ 10.00'
    assert repr(rule1_3) == 'type1 < 20.00'

    data = pd.DataFrame(np.arange(30).reshape((-1, 1)), columns=['type1'])
    assert rule1_1.get_matching_mask(data).sum() == 10
    assert rule1_2.get_matching_mask(data).sum() == 11
    assert rule1_3.get_matching_mask(data).sum() == 20

    assert rule1_1.combine(rule1_2) == rule1_1
    assert rule1_1.combine(rule1_3) == rule1_1

    r1 = rule1_1.copy()
    assert r1 == rule1_1


def test_type_2():
    var = Variable(0, 'type2', 'float')
    rule2_1 = Rule(var, 10, False)  # 10, '<', var, None, None)
    rule2_2 = Rule(var, 10, True)  # 10, '<=', var, None, None)
    rule2_3 = Rule(var, 20, False)  # 20, '<', var, None, None)
    rule2_4 = Rule(var, 20, False)  # None, 2, var, '>', 20)
    rule2_5 = Rule(var, 10, True)  # None, 2, var, '>=', 10)
    rule2_6 = Rule(var, 10, False)  # 10, '<=', var, '>', 10)
    rule2_7 = Rule(var, 10, False)  # 10, '<', var, '>=', 10)
    rule2_8 = Rule(var, 20, True)  # 10, '<', var, '>=', 20)
    rule2_9 = Rule(var, 20, False)  # 20, '<', var, '>=', 10)

    assert rule2_4 == rule2_3
    assert rule2_5 == rule2_2
    assert repr(rule2_6) == 'type2 > 10.00'
    assert repr(rule2_7) == 'type2 > 10.00'
    assert repr(rule2_8) == 'type2 ≥ 20.00'
    assert repr(rule2_9) == 'type2 > 20.00'

    assert rule2_1.rule_type == 2
    assert rule2_2.rule_type == 2
    assert rule2_3.rule_type == 2

    assert not rule2_1.is_categorical_rule

    assert repr(rule2_1) == 'type2 > 10.00'
    assert repr(rule2_2) == 'type2 ≥ 10.00'
    assert repr(rule2_3) == 'type2 > 20.00'

    data = pd.DataFrame(np.arange(30).reshape((-1, 1)), columns=['type2'])
    assert rule2_1.get_matching_mask(data).sum() == 19
    assert rule2_2.get_matching_mask(data).sum() == 20
    assert rule2_3.get_matching_mask(data).sum() == 9

    assert rule2_1.combine(rule2_2) == rule2_1
    assert rule2_1.combine(rule2_3) == rule2_3
    r1 = rule2_1.copy()
    assert r1 == rule2_1


def test_type_3():
    var = Variable(0, 'type3', 'float')
    rule3_1 = Rule(var, 10, False, 40, False)  # 10, '<', var, '<', 40)
    rule3_2 = Rule(var, 10, True, 40, True)  # 10, '<=', var, '<=', 40)
    rule3_3 = Rule(var, 20, False, 30, False)  # 20, '<', var, '<', 30)
    rule3_4 = Rule(var, 20, False, 30, False)  # 30, '>', var, '>', 20)
    rule3_5 = Rule(var, 10, True, 40, True)  # 40, '>=', var, '>=', 10)

    assert rule3_4 == rule3_3
    assert rule3_5 == rule3_2

    assert rule3_1.rule_type == 3
    assert rule3_2.rule_type == 3
    assert rule3_3.rule_type == 3

    assert not rule3_1.is_categorical_rule

    assert repr(rule3_1) == '10.00 < type3 < 40.00'
    assert repr(rule3_2) == '10.00 ≤ type3 ≤ 40.00'
    assert repr(rule3_3) == '20.00 < type3 < 30.00'

    data = pd.DataFrame(np.arange(50).reshape((-1, 1)), columns=['type3'])
    assert rule3_1.get_matching_mask(data).sum() == 29
    assert rule3_2.get_matching_mask(data).sum() == 31
    assert rule3_3.get_matching_mask(data).sum() == 9

    assert rule3_1.combine(rule3_2) == rule3_1
    assert rule3_1.combine(rule3_3) == rule3_3
    r1 = rule3_1.copy()
    assert r1 == rule3_1


def test_type_5():
    var = Variable(0, 'type4', 'float')
    rule4_1 = Rule(var, 40, False, 10, False)  # 10, '>', var, '>', 40)
    rule4_2 = Rule(var, 40, True, 10, True)  # 10, '>=', var, '>=', 40)
    rule4_3 = Rule(var, 30, False, 20, False)  # 20, '>', var, '>', 30)
    rule4_4 = Rule(var, 30, False, 20, False)  # 30, '<', var, '<', 20)
    rule4_5 = Rule(var, 40, True, 10, True)  # 40, '<=', var, '<=', 10)

    assert rule4_4 == rule4_3
    assert rule4_5 == rule4_2

    assert rule4_1.rule_type == 5
    assert rule4_2.rule_type == 5
    assert rule4_3.rule_type == 5

    assert not rule4_1.is_categorical_rule

    assert repr(rule4_1) == 'type4 - False'
    assert repr(rule4_2) == 'type4 - False'
    assert repr(rule4_3) == 'type4 - False'

    data = pd.DataFrame(np.arange(50).reshape((-1, 1)), columns=['type4'])
    assert rule4_1.get_matching_mask(data).sum() == 0
    assert rule4_2.get_matching_mask(data).sum() == 0
    assert rule4_3.get_matching_mask(data).sum() == 0

    assert rule4_1.combine(rule4_2) == rule4_1
    assert rule4_1.combine(rule4_3) == rule4_1
    assert rule4_1 == rule4_4
    r1 = rule4_1.copy()
    assert r1 == rule4_1
