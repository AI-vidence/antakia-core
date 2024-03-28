from unittest import TestCase

import pandas as pd
from skimage.metrics import mean_squared_error

from antakia_core.compute.model_subtitution.model_interface import InterpretableModels
from antakia_core.data_handler import Region, Rule, RuleSet, RegionSet, ModelRegion, ModelRegionSet
from antakia_core.utils import Variable
from antakia_core.utils.utils import int_mask_to_boolean, ProblemCategory
from antakia_core.compute.model_subtitution.regression_models import *
from tests.utils_fct import dummy_mask
import numpy as np


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
        self.mask = dummy_mask(self.X)

    def test_init(self):
        """
        Tests that the region is correctly instanciated,
        wether a mask or a rule is given or not

        Tests the implementation of property and setter : color, name

        """
        rule_set = RuleSet([self.r1_1])

        # init a region with no mask and no ruleset
        region = Region(self.X)
        assert region.mask.equals(pd.Series([False] * len(self.X), index=self.X.index))
        # init a rule with a mask
        region = Region(self.X, mask=self.mask)
        assert region.mask.equals(self.mask)
        # init a region with a ruleset and no mask
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

        # test the name of a leftouf region
        region2 = RegionSet(self.X)._compute_left_out_region()
        assert region2.name == 'left outs'

        # test the name of a autoclustered region with and without rule set
        region3 = Region(self.X)
        region3.auto_cluster = True
        assert region3.name == 'auto-cluster'
        region3 = Region(self.X, rule_set)
        region3.auto_cluster = True
        assert region3.name == 'AC: 2.00 ≤ var1 < 10.00'

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
        rule_set = RuleSet([self.r1_1])
        region = Region(self.X, rule_set)
        assert isinstance(region.get_color_serie(), pd.Series)
        region.color = 'grey'
        assert 'blue' in list(region.get_color_serie())


class TestModelRegion(TestCase):

    def setUp(self):
        np.random.seed(1234)
        X = np.random.randn(500, 4)
        y = np.sum(X, axis=1)
        self.X_train = pd.DataFrame(X[:250], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_train = pd.Series(y[:250])

        # X, y = generate_corner_dataset(100, random_seed=1234)
        self.X_test = pd.DataFrame(X[250:], columns=['var1', 'var2', 'var3', 'var4'])
        self.y_test = pd.Series(y[250:])
        # = generate_corner_dataset(100, random_seed=4321)

        self.customer_model = None
        self.mask = int_mask_to_boolean(dummy_mask(self.y_train, random_seed=14))
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

        self.r1_3 = Rule(self.v1,
                         min=0)

    def test_init(self):
        # init
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        self.customer_model = ModReg.interpretable_models.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)

        assert ModReg.X.equals(self.X_train)
        assert ModReg.y.equals(self.y_train)
        assert ModReg._test_mask is None

        # perfs
        # when no perf computed
        assert ModReg.perfs.equals(pd.DataFrame())
        ModReg.interpretable_models.get_models_performance(self.customer_model, self.X_train, self.y_train, self.X_test,
                                                           self.y_test)
        assert ModReg.perfs.shape[0] != 0

        # delta
        assert ModReg.delta == 0
        ModReg.interpretable_models.select_model('Linear Regression')
        assert ModReg.delta == 0

        # test_mask property
        assert not ModReg.test_mask.any()
        ModReg.rules.add(self.r1_1)
        assert ModReg.test_mask.any()

        ModReg1 = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                              self.customer_model, score='mse')
        ModReg1.interpretable_models._init_models(ProblemCategory.regression)
        self.customer_model = ModReg1.interpretable_models.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)

        ModReg1.rules.add(self.r1_3)
        ModReg1.update_mask(ModReg1.rules.get_matching_mask(self.X_test))
        assert ModReg1.test_mask.any()

    def test_train_residuals(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='MSE')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        self.customer_model = ModReg.interpretable_models.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)
        ModReg.interpretable_models._init_scores(self.customer_model, ProblemCategory.regression, X_test=self.X_test,
                                                 y_test=self.y_test)
        ModReg.interpretable_models.get_models_performance(self.customer_model, self.X_train, self.y_train, self.X_test,
                                                           self.y_test)
        assert len(ModReg.train_residuals('Linear Regression')) == ModReg.mask.astype(int).sum()

    def test_to_dict(self):
        # test with initialization with a mask
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='MSE')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        self.customer_model = ModReg.interpretable_models.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)
        ModReg.interpretable_models._init_scores(self.customer_model, ProblemCategory.regression, X_test=self.X_test,
                                                 y_test=self.y_test)
        ModReg.interpretable_models.get_models_performance(self.customer_model, self.X_train, self.y_train, self.X_test,
                                                           self.y_test)
        ModReg.interpretable_models.select_model('Linear Regression')
        assert ModReg.to_dict() == {
            'Region': -1,
            'Rules': '',
            'Average': '-2.44e-03',
            'Points': 130,
            '% dataset': '52.0%',
            'Sub-model': 'LR - MSE:0.00 (0.00)',
            'color': 'cyan'
        }

    def test_select_model(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='mse')
        ModReg.select_model('model_selected')
        assert ModReg.interpretable_models.selected_model == 'model_selected'

    def test_train_substitution_model(self):
        # init
        int_mod = InterpretableModels(mean_squared_error)
        int_mod._init_models(task_type=ProblemCategory.regression)
        self.customer_model = int_mod.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='MSE')
        ModReg.interpretable_models._init_scores(self.customer_model, ProblemCategory.regression, X_test=self.X_test,
                                                 y_test=self.y_test)

        # test when X_test is None
        ModReg.train_substitution_models(ProblemCategory.regression)
        # test when X_test is not None
        ModReg = ModelRegion(self.X_train, self.y_train, None, None,
                             self.customer_model, mask=self.mask, score='MSE')
        ModReg.train_substitution_models(ProblemCategory.regression)

    def test_get_model(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert ModReg.get_model('Linear Regression').name == LinearRegression().name

    def test_get_selected_model(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        # checks that the functions returns None when no model is selected
        assert ModReg.get_selected_model() is None
        # checks that the model returns the selected model
        ModReg.select_model('Linear Regression')
        assert ModReg.get_selected_model().name == LinearRegression().name

    def test_predict(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='MSE')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        self.customer_model = ModReg.interpretable_models.models['Linear Regression']
        self.customer_model.fit(self.X_train, self.y_train)
        ModReg.interpretable_models._init_scores(self.customer_model, ProblemCategory.regression, X_test=self.X_test,
                                                 y_test=self.y_test)
        ModReg.interpretable_models.get_models_performance(self.customer_model, self.X_train, self.y_train, self.X_test,
                                                           self.y_test)

        # check that y_pred is a pd.Series of NaN when no model is selected
        y_pred = ModReg.predict(self.X_train)
        assert isinstance(y_pred, pd.Series)
        assert y_pred.isna().all()

        # test when a model is selected
        ModReg.select_model('Linear Regression')
        y_pred = ModReg.predict(self.X_train)
        assert isinstance(y_pred, pd.Series)

    def test_update_rule_set(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert not ModReg.interpretable_models.models == {}
        ModReg.rules.add(self.r1_1)
        ModReg.update_rule_set(RuleSet([self.r1_2]))
        # check that when updated, the rule set contains the new rules
        assert repr(ModReg.rules) == repr(RuleSet([self.r1_2]))
        # checks that interpretable_models is reset
        assert ModReg.interpretable_models.models == {}

    def test_update_mask(self):
        ModReg = ModelRegion(self.X_train, self.y_train, self.X_test, self.y_test,
                             self.customer_model, mask=self.mask, score='mse')
        ModReg.interpretable_models._init_models(ProblemCategory.regression)
        assert not ModReg.interpretable_models.models == {}
        ModReg.mask = self.r1_1.get_matching_mask(self.X_train)
        ModReg.update_mask(self.r1_2.get_matching_mask(self.X_train))
        # check that when updated, the mask attribute contains the new mask, and rule set is empty
        assert ModReg.mask.equals(self.r1_2.get_matching_mask(self.X_train))
        assert len(ModReg.rules) == 0
        # checks that interpretable_models is reset
        assert ModReg.interpretable_models.models == {}
