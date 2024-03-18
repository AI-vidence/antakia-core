import numpy as np
import pandas as pd

from antakia_core.data_handler.rules import Rule, Variable


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


def test_combine():
    var1 = Variable(0, 'comb1', 'float')
    rule1_1 = Rule(var1, max=20,
                   includes_max=False)  # None, None, var1, '<', 20)
    rule1_2 = Rule(var1, max=10,
                   includes_max=False)  # None, None, var1, '<', 10)
    rule1_3 = Rule(var1, max=10,
                   includes_max=True)  # None, None, var1, '<=', 10)
    rule1_4 = Rule(var1, max=5,
                   includes_max=False)  # None, None, var1, '<', 5)
    rule2_1 = Rule(
        var1,
        min=10,
        includes_min=True,
    )  # 10, '<=', var1, None, None)
    rule3_1 = Rule(var1, min=10, includes_min=True, max=40,
                   includes_max=False)  # 10, '<=', var1, '<', 40)
    rule4_1 = Rule(var1,
                   min=40,
                   includes_min=False,
                   max=10,
                   includes_max=False)  # 10, '>', var1, '>', 40)

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
