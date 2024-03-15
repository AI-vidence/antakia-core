import pandas as pd

from antakia_core.data_handler.rules import Rule, RuleSet
from antakia_core.data_handler.region import RegionSet, Region
from antakia_core.utils.variable import Variable


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
    rule1 = Rule(var, min=2, includes_min=True, max=10, includes_max=False)

    region = rs.add_region(rules=RuleSet([rule1]))
    assert rs.get(1) == region
    assert len(rs) == 1
    assert rs.get_max_num() == 1
    assert rs.get_new_num() == 2

    color = rs.get_color_serie()
    assert (color == pd.Series(['grey', 'red', 'red', 'grey', 'grey'])).all()

    rs.clear_unvalidated()
    rs.add_region(RuleSet([rule1]), color='blue')
    assert rs.get_max_num() == 1
    assert rs.get_new_num() == 2
    assert rs.get(1).color == 'blue'

    color = rs.get_color_serie()
    assert (color == pd.Series(['grey', 'blue', 'blue', 'grey', 'grey'])).all()

    var2 = Variable(0, 'var2', 'float')
    rule3 = Rule(var2, min=1.5, includes_min=False)
    rs.add_region(RuleSet([rule3]), color='blue')
    rs.stats()

    r = Region(data, RuleSet([rule3]))
    r.num = -1

    rs.add(r)
    assert r.num == 3
