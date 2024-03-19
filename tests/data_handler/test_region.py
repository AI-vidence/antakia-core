from unittest import TestCase

import pandas as pd

from src.antakia_core.data_handler.rules import Rule, RuleSet
from src.antakia_core.data_handler.region import RegionSet, Region, ModelRegion
from src.antakia_core.utils.variable import Variable
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask


class TestRegion(TestCase):
    def setUp(self):
        pass

    def test_regions(self):
        """
        Tests that the region is correctly instanciated,
        wether a mask or a rule is given or not

        Tests the implementation of property and setter : color, name

        """
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
        rule1 = Rule(None, None, var, '<', 10)
        rule2 = Rule(2, '<=', var, None, None)

        region = rs.add_region(rules=RuleSet([rule1, rule2]))
        assert rs.get(1) == region
        assert len(rs) == 1
        assert rs.get_max_num() == 1
        assert rs.get_new_num() == 2

        color = rs.get_color_serie()
        assert (color == pd.Series(['grey', 'red', 'red', 'grey', 'grey'])).all()

        rs.clear_unvalidated()
        rs.add_region(RuleSet([rule1, rule2]), color='blue')
        assert rs.get_max_num() == 1
        assert rs.get_new_num() == 2
        assert rs.get(1).color == 'blue'

        color = rs.get_color_serie()
        assert (color == pd.Series(['grey', 'blue', 'blue', 'grey', 'grey'])).all()

        var2 = Variable(0, 'var2', 'float')
        rule3 = Rule(None, None, var2, '>', 1.5)
        rs.add_region(RuleSet([rule3]), color='blue')
        rs.stats()

        r = Region(data, RuleSet([rule3]))
        r.num = 1

        rs.add(r)
        assert r.num == 3

        assert region.to_dict() == {'% dataset': '40.0%', 'Average': None, 'Points': 2, 'Region': 1,
                                    'Rules': 'var1 < 10.00 and var1 ≥ 2.00', 'Sub-model': None, 'color': 'red'}
        assert region.num_points() == 2
        assert region.dataset_cov() == 0.4
        region.validate()
        assert region.validated
        assert region.name == 'var1 < 10.00 and var1 ≥ 2.00'
        region.color = 'Blue'
        assert region.color == 'Blue'


class TestModelRegion(TestCase):
    def setUp(self):
        self.X = pd.DataFrame(generate_corner_dataset(10)[0])
        self.y = pd.DataFrame(generate_corner_dataset(10)[1])
        self.X_test = pd.DataFrame(generate_corner_dataset(10)[0])
        self.y_test = pd.DataFrame(generate_corner_dataset(10)[1])
        # self.test_mask = None
        # self.costumer_model =

    def test_init(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test, self.costumer_model)

    def test_to_dict(self):
        assert ModReg.to_dict().equals()


class TestRegionSet(TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            [1, 2],
            [2, 1],
            [4, 2],
            [10, 1],
            [20, 2],
        ], columns=['var1', 'var2'])
        self.mask = dummy_mask(self.data)

    def test_init_properties(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.mask, pd.Series)
        assert len(rs) == len(rs.regions)
    def test_get_max_num(self):
        rs = RegionSet(self.data)
        assert rs.get_max_num() == 0
        var1 = Variable(0, 'var1', 'float')
        rule1 = Rule(None, None, var1, '>', 1.5)
        region = Region(self.data, RuleSet([rule1]))
        rs.add(region)
        assert rs.get_max_num() == 0


    def test_add_region(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        added_region = rs.add_region(mask=self.mask)
        assert isinstance(added_region, Region)

    def test_extend(self):
        rs = RegionSet(self.data)
        rs_added = RegionSet(self.data)
        rs_added.add_region(mask=self.mask)
        rs.extend(rs_added)

    def test_remove(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        rs.remove(1)

    def test_to_dict(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=dummy_mask(self.data, 10))
        assert isinstance(rs.to_dict(), list)
        if not len(rs.to_dict()):
            assert isinstance(rs.to_dict()[0], dict)

    def test_get_colors(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.get_colors(), list)
        if not len(rs.get_colors()):
            assert isinstance(rs.get_colors()[0], str)

    def test_get_masks(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.get_masks(), list)
        if not len(rs.get_masks()):
            assert isinstance(rs.get_masks()[0], pd.Series)

    def test_get_color_series(self):
        rs = RegionSet(self.data)
        rs.add_region(mask=self.mask)
        assert isinstance(rs.get_color_serie(), pd.Series)

class TestModelRegionSet(TestCase):
    def setUp(self):
        pass

    def test_init(self):
        pass