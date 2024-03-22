from unittest import TestCase

import pandas as pd

from antakia_core.data_handler import Region, Rule, RuleSet, RegionSet, ModelRegion, ModelRegionSet
from antakia_core.utils import Variable
from antakia_core.utils.utils import int_mask_to_boolean, ProblemCategory
from antakia_core.compute.model_subtitution.classification_models import *
from antakia_core.compute.model_subtitution.regression_models import *
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask, DummyModel


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
        X, y = generate_corner_dataset(10, random_seed=1234)
        X_test, y_test = generate_corner_dataset(10, random_seed=4321)

        self.X = pd.DataFrame(X, columns=['var1','var2'])
        self.y = pd.Series(y)
        self.X_test = pd.DataFrame(X_test)
        self.y_test = pd.Series(y_test)
        self.costumer_model = DummyModel()
        self.mask = int_mask_to_boolean(dummy_mask(self.y,random_seed=14))
        self.v1 = Variable(0, 'var1', 'float')
        self.r1_1 = Rule(self.v1,
                         min=2,
                         includes_min=True,
                         max=10,
                         includes_max=False)

        self.r1_2 = Rule(self.v1,
                         min=0.2,
                         includes_min=True,
                         max=0.8,
                         includes_max=False)

    def test_init(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, score='mse')

    def test_to_dict(self):
        #test with initialization with a mask
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='MSE')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        ModReg.select_model('Linear Regression')
        assert ModReg.to_dict() == {
            'Region': -1,
            'Rules': '',
            'Average': '0.33',
            'Points': 3,
            '% dataset': '30.0%',
            'Sub-model': 'Linear Regression',
            'color': 'cyan'
        }

        #test with initialization with a falsy rule and a selected model
        # ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
        #                      self.costumer_model, rules=RuleSet([self.r1_1]), score='mse')
        # assert ModReg.to_dict() == {
        #     'Region': -1,
        #     'Rules': '2.00 ≤ var1 < 10.00',
        #     'Average': 'NaN',
        #     'Points': 0,
        #     '% dataset': '0.0%',
        #     'Sub-model': None,
        #     'color': 'cyan'
        # }
        #
        # #test with initialization with an interval rule
        # ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
        #                      self.costumer_model, rules=RuleSet([self.r1_2]), score='mse')
        # assert ModReg.to_dict() == {
        #     'Region': -1,
        #     'Rules': '0.20 ≤ var1 < 0.80',
        #     'Average': '0.25',
        #     'Points': 8,
        #     '% dataset': '80.0%',
        #     'Sub-model': None,
        #     'color': 'cyan'
        # }
    def test_select_model(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.select_model('model_selected')
        assert ModReg.interpretable_models.selected_model == 'model_selected'

    def test_train_substitution_model(self):  # not ok
        #test when X_test is None
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')

        #test when X_test is not None
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')

    def test_get_model(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert ModReg.get_model('Linear Regression').name == LinearRegression().name
    def test_get_selected_model(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        #checks that the functions returns None when no model is selected
        assert ModReg.get_selected_model() is None
        #checks that the model returns the selected model
        ModReg.select_model('Linear Regression')
        assert ModReg.get_selected_model().name == LinearRegression().name

    def test_predict(self):  # not ok
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)

        #check that y_pred is a pd.Series of NaN when no model is selected
        # y_pred = ModReg.predict(self.X)
        # assert isinstance(y_pred, pd.Series)
        # assert y_pred.isna().all()

        # test when a model is selected
        model_fitted = LinearRegression()
        model_fitted.fit(X = self.X, y = self.y)
        ModReg.interpretable_models.select_model(model_fitted)
        # ModReg.interpretable_models.selected_model = ModReg.interpretable_models.selected_model.fit()
        y_pred = ModReg.predict(self.X)
        assert isinstance(y_pred, pd.Series)
        assert not y_pred.isna().all()



    def test_update_rule_set(self):
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert not ModReg.interpretable_models.models == {}
        ModReg.rules.add(self.r1_1)
        ModReg.update_rule_set(RuleSet([self.r1_2]))
        #check that when updated, the rule set contains the new rules
        assert repr(ModReg.rules) == repr(RuleSet([self.r1_2]))
        #checks that interpretable_models is reset
        assert ModReg.interpretable_models.models == {}

    def test_update_mask(self):  # not ok
        ModReg = ModelRegion(self.X, self.y, self.X_test, self.y_test,
                             self.costumer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert not ModReg.interpretable_models.models == {}
        ModReg.mask = self.r1_1.get_matching_mask(self.X)
        ModReg.update_mask(self.r1_2.get_matching_mask(self.X))
        # check that when updated, the mask attribute contains the new mask, and rule set is empty
        assert ModReg.mask.equals(self.r1_2.get_matching_mask(self.X))
        assert len(ModReg.rules) == 0
        # checks that interpretable_models is reset
        assert ModReg.interpretable_models.models == {}

