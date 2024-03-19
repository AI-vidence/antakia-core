from unittest import TestCase
import pandas as pd

from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.utils.variable import Variable, DataVariables
from antakia_core.data_handler.rules import Rule, RuleSet
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask


class TestSkopeRule(TestCase):
    def setUp(self):
        pass

    def test_skope_rules(self):
        df = pd.DataFrame(generate_corner_dataset(10)[0])
        mask = dummy_mask(df)
        var1 = Variable(0, 'type1', 'float')
        var2 = Variable(0, 'type2', 'float')
        variables = DataVariables([var1,var2])

        # skope_rules(mask, df, variables)
