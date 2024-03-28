from unittest import TestCase

import pandas as pd
import numpy as np

from antakia_core.data_handler import Rule, Region, RegionSet, RuleSet, ModelRegionSet, ModelRegion
from antakia_core.data_handler.rule import FalsyRule, TruthyRule
from antakia_core.utils import Variable, ProblemCategory
from tests.utils_fct import DummyModel, dummy_mask


class TestRegionSet(TestCase):

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
        self.r1_2 = Rule(self.v1, min=2, includes_min=True)
        self.rs1 = RuleSet([self.r1_1, self.r1_2])
        self.r2_1 = Rule(self.v2, min=1.5, includes_min=False)

        self.region_set = RegionSet(self.X)

        self.mask = dummy_mask(self.X)

    def test_init(self):
        rs = RegionSet(self.X)
        assert len(rs) == 0
        assert rs.X is self.X
        assert isinstance(rs.left_out_region, Region)
        assert len(rs.display_order) == 0

    def test_get_new_num(self):
        # test for empty rs
        rs = RegionSet(self.X)
        assert rs.get_new_num() == 1

        # test for rs
        rs.add(Region(self.X, RuleSet([self.r1_1])))
        rs.add(Region(self.X, RuleSet([self.r1_2])))
        rs.add(Region(self.X, RuleSet([self.r2_1])))
        assert rs.get_new_num() == 4

        # test when region nums are 1 and 3
        rs.remove(2)
        assert rs.get_new_num() == 2

    def test_get_max_num(self):
        # test for empty rs
        rs = RegionSet(self.X)
        assert rs.get_max_num() == 0

        # test for not empty rs
        rs.add(Region(self.X, RuleSet([self.r1_1])))
        rs.add(Region(self.X, RuleSet([self.r2_1])))
        assert rs.get_max_num() == 2

        # set manually max to 5
        rs = RegionSet(self.X)
        region = Region(self.X, RuleSet([self.r1_1]))
        region.num = 5
        rs.add(region)
        rs.add(Region(self.X, RuleSet([self.r2_1])))
        assert rs.get_max_num() == 5

    def test_add(self):
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        region = Region(self.X, RuleSet([self.r1_2]))
        region.num = 1
        rs.add(region)
        assert len(rs) == 1

    def test_add_region(self):
        # add with a mask
        rs = RegionSet(self.X)
        added_region = rs.add_region(mask=self.mask, auto_cluster=True)
        assert isinstance(added_region, Region)
        assert added_region.auto_cluster

        # add with a rule set
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1, self.r1_2]))

        rs1 = RegionSet(self.X)
        rs1.add_region(rules=RuleSet([self.r1_2]))

        rs2 = RegionSet(self.X)
        rs2.add_region(rules=RuleSet([self.r1_1]))

        assert (RegionSet(self.X).add_region(rules=RuleSet([self.r1_1, self.r1_2])).mask.equals(
            rs.mask))

    def test_extend(self):
        # extend empty with empty gives empty
        rs = RegionSet(self.X)
        rs.extend(rs)
        assert len(rs) == 0

        rs_added = RegionSet(self.X)
        rs_added.add_region(mask=self.mask)
        rs.add_region(rules=RuleSet([self.r1_1, self.r1_2]))
        rs.extend(rs_added)
        assert len(rs) == 2

    def test_remove(self):
        rs = RegionSet(self.X)
        rs.add_region(mask=self.mask)
        rs.remove(1)
        assert len(rs) == 0

        # ajouter test avec leftout region

    def test_to_dict(self):
        rs = RegionSet(self.X)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.to_dict(), list)
        assert isinstance(rs.to_dict()[0], dict)

    def test_get_masks(self):  # not ok
        # test empty region set
        rs = RegionSet(self.X)
        assert rs.get_masks() == []

        rs.add_region(mask=self.mask)
        assert rs.get_masks() == [rs.regions[1].mask]

    def test_get_colors(self):
        rs = RegionSet(self.X)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.get_colors(), list)
        assert len(rs.get_colors()) == len(rs.display_order)
        if not len(rs.get_colors()) == 0:
            assert isinstance(rs.get_colors()[0], str)

    def test_get_color_series(self):
        # test signature
        rs = RegionSet(self.X)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.get_color_serie(), pd.Series)

        color = rs.get_color_serie()
        assert (color == pd.Series(['red', 'red', 'grey',
                                    'grey', 'grey'])).all()

    def test_get(self):
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        assert repr(rs.get(1).rules) == repr(Region(self.X, rules=RuleSet([self.r1_1])).rules)
        assert rs.get(1).mask.equals(Region(self.X, rules=RuleSet([self.r1_1])).mask)

        rs2 = RegionSet(self.X)
        assert rs2.get('-').name == 'left outs'

    def test_clear_unvalidated(self):
        # create a rule set of three rules, validate 2, check that only one was cleared
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        rs.add_region(rules=RuleSet([self.r1_2]))
        rs.regions[2].validate()
        rs.add_region(rules=RuleSet([self.r2_1]))
        rs.regions[3].validate()
        rs.clear_unvalidated()
        assert len(rs) == 2

    def test_pop_last(self):
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        rs.add_region(rules=RuleSet([self.r1_2]))
        rs.add_region(rules=RuleSet([self.r2_1]))
        rs.regions[1].validate()

        # check that the returned region is the last one
        assert rs.pop_last().mask.equals(Region(self.X, rules=RuleSet([self.r2_1])).mask)
        assert len(rs) == 2
        # check that last region is removed from region set
        assert rs.pop_last().mask.equals(Region(self.X, rules=RuleSet([self.r1_2])).mask)
        assert len(rs) == 1
        # check that validated regions are not removed but yet returned
        assert isinstance(rs.pop_last(), Region)
        assert len(rs) == 1
        # check that the region is removed once unvalidated
        rs.regions[1].validated = False
        rs.pop_last()
        # checks that removing from empty region set returns None
        assert rs.pop_last() is None

    def test_sort(self):
        # create 3 regions, sort by id, size, insert order and check the order in the rs
        r1_1 = Rule(self.v1, max=4, includes_max=False)
        r1_2 = Rule(self.v1, min=2, includes_min=True)
        r2_1 = Rule(self.v2, min=1.5, includes_min=False)

        rs = RegionSet(self.X)
        region1 = Region(self.X, rules=RuleSet([r1_1]))
        region2 = Region(self.X, rules=RuleSet([r1_2]))
        region2.num = 3
        region3 = Region(self.X, rules=RuleSet([r2_1]))
        rs.add(region1)
        rs.add(region2)
        rs.add(region3)
        assert rs.display_order == [region1, region2, region3]
        rs.sort('region_num')
        assert rs.display_order == [region1, region3, region2]
        rs.sort('region_num', ascending=False)
        assert rs.display_order == [region2, region3, region1]
        rs.sort('size')
        assert rs.display_order == [region1, region3, region2]
        rs.sort('insert')
        assert rs.display_order == [region1, region2, region3]

    def test_stats(self):
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        assert rs.stats()['regions'] == 1
        rs.add_region(rules=RuleSet([self.r1_2]))
        assert rs.stats()['points'] == 5
        rs.add_region(rules=RuleSet([self.r2_1]))
        assert rs.stats()['coverage'] == 100

    def test_compute_left_out_region(self):
        # test if combining mask and leftout mask gives
        rs = RegionSet(self.X)
        rs.add_region(rules=RuleSet([self.r1_1]))
        assert not np.array(
            [rs._compute_left_out_region().mask[i] and rs.mask[i] for i in range(self.X.shape[0])]).all()

        # tester falsy rule
        rs1 = RegionSet(self.X)
        rs1.add_region(rules=RuleSet([FalsyRule(self.v1)]))
        assert rs1._compute_left_out_region().mask.all()

        # tester truthy rule
        rs2 = RegionSet(self.X)
        rs2.add_region(rules=RuleSet([TruthyRule(self.v1)]))
        assert not rs2._compute_left_out_region().mask.all()


class TestModelRegionSet(TestCase):

    def setUp(self):
        np.random.seed(1234)
        X = np.random.randn(500, 4)
        y = np.sum(X, axis=1)
        self.X_train = pd.DataFrame(X[:250], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_train = pd.Series(y[:250])

        self.X_test = pd.DataFrame(X[250:], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_test = pd.Series(y[250:])

        self.customer_model = None
        self.mask = dummy_mask(self.y_train, random_seed=14).astype(bool)
        self.v1 = Variable(0, 'var1', 'float')
        self.rule1_1 = Rule(self.v1,
                            min=2,
                            includes_min=True,
                            max=10,
                            includes_max=False)

        self.rule1_2 = Rule(self.v1,
                            min=0.2,
                            includes_min=True,
                            max=0.8,
                            includes_max=False)

        self.rule1_3 = Rule(self.v1,
                            min=0)
        self.rules = [self.rule1_1, self.rule1_2, self.rule1_3]

        self.problem_category = ProblemCategory.regression

    def test_init(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        assert mod_region_set.model == self.customer_model
        assert mod_region_set.X_test.equals(self.X_test)
        assert mod_region_set.X.equals(self.X_train)

    def test_upgrade_region_to_model_region(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)

        region = Region(self.X_train, RuleSet(self.rules))
        region.validate()
        ModReg = mod_region_set.upgrade_region_to_model_region(region)
        assert ModReg.validated

    def test_add(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)

        region = Region(self.X_train, RuleSet([self.rule1_2]))
        # test adding a Region object
        mod_region_set.add(region)
        assert len(mod_region_set) == 1
        # test adding a ModelRegion object
        mod_region = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                                 self.customer_model, score='mse')
        mod_region_set.add(mod_region)
        assert len(mod_region_set) == 2

    def test_add_region(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        # add with a mask
        added_region = mod_region_set.add_region(mask=self.mask, auto_cluster=True)
        assert isinstance(added_region, ModelRegion)
        assert added_region.auto_cluster
        assert len(mod_region_set.regions) == 1

        # add with a rule set
        mod_region_set.add_region(rules=RuleSet(self.rules))
        assert len(mod_region_set.regions) == 2

    def test_get(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        mod_region_set.add_region(rules=RuleSet(self.rules))
        assert isinstance(mod_region_set.get(1), ModelRegion)
        assert repr(mod_region_set.get(1).rules) == repr(Region(self.X_train, rules=RuleSet(self.rules)).rules)
        assert mod_region_set.get(1).mask.equals(Region(self.X_train, rules=RuleSet(self.rules)).mask)

        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        assert mod_region_set.get('-').name == 'left outs'

    def test_stats(self):
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        mod_region_set.add_region(rules=RuleSet([self.rule1_1]))
        assert isinstance(mod_region_set.stats(), dict)
        assert mod_region_set.stats()['regions'] == 1
        mod_region_set.add_region(rules=RuleSet([self.rule1_2]))
        assert mod_region_set.stats()['points'] == 69
        mod_region_set.add_region(rules=RuleSet([self.rule1_3]))
        assert mod_region_set.stats()['coverage'] == 54
        assert mod_region_set.stats()['delta_score'] == 0.0

    def test_predict(self):  # not ok
        mod_region_set = ModelRegionSet(self.X_train, self.y_train, self.X_test,
                                        self.y_test, self.customer_model,
                                        lambda *args: 1)
        assert mod_region_set.predict(self.X_train).equals(pd.Series(index=self.X_train.index))
        mod_region_set.add_region(rules=RuleSet([self.rule1_1]))
        assert mod_region_set.predict(self.X_train).shape == self.y_train.shape
