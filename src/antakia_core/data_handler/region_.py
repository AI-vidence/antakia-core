from __future__ import annotations

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from antakia_core.compute.model_subtitution.model_interface import InterpretableModels
from antakia_core.data_handler.rules import RuleSet
from antakia_core.utils.utils import colors, boolean_mask, format_number, BASE_COLOR


class Region:
    """
    class to handle regions
    a region is defined either by a selection of point or by a set of rules
    """
    region_colors = colors
    LEFT_OUT_NUM = '-'

    def __init__(self,
                 X,
                 rules: RuleSet | None = None,
                 mask: pd.Series | None = None,
                 color=None,
                 num=-1):
        """

        Parameters
        ----------
        X : base dataframe to use for the rule
        rules : list of rules
        mask : selected points
        color: region color, if not provided, auto assigned
        """
        self.X = X
        self.num = num
        self.rules = RuleSet(rules)
        if mask is None:
            # if no mask, compute it
            if rules is not None:
                self.mask = self.rules.get_matching_mask(X)
            else:
                self.mask = pd.Series([False] * len(X), index=X.index)
        else:
            self.mask = mask
        self._color = color
        self.validated = False
        self.auto_cluster = False

    @property
    def color(self):
        """
        get region color
        Returns
        -------

        """
        if self._color is None:
            return self.region_colors[(self.num - 1) % len(self.region_colors)]
        return self._color

    @color.setter
    def color(self, c):
        """
        set region color
        Parameters
        ----------
        c

        Returns
        -------

        """
        self._color = c

    @property
    def name(self):
        """
        get region name
        Returns
        -------

        """
        if self.num == self.LEFT_OUT_NUM:
            return 'left outs'
        name = repr(self.rules)
        if self.auto_cluster:
            if name:
                name = 'AC: ' + name
            else:
                name = "auto-cluster"
        return name

    def to_dict(self) -> dict[str, str | int | None]:
        """
        get region as dict
        Returns
        -------

        """
        dict_form = {
            "Region": self.num,
            "Rules": self.name,
            "Average": None,
            "Points": self.mask.sum(),
            "% dataset": f"{round(self.mask.mean() * 100, 2)}%",
            "Sub-model": None,
            "color": self.color
        }
        return dict_form

    def num_points(self) -> int:
        """
        get the number of points on the region
        Returns
        -------

        """
        return self.mask.sum()

    def dataset_cov(self):
        """
        get Region's dataset coverage (% of points in the Region)
        Returns
        -------

        """
        return self.mask.mean()

    def validate(self):
        """
        set Region as validated
        will not be erased by auto clustering
        Returns
        -------

        """
        self.validated = True

    def update_rule_set(self, rule_set: RuleSet):
        self.rules = rule_set
        self.mask = self.rules.get_matching_mask(self.X)
        self.validated = False

    def update_mask(self, mask: pd.Series):
        self.mask = mask
        self.rules = RuleSet()

    def get_color_serie(self):
        color = pd.Series([BASE_COLOR] * len(self.X), index=self.X.index)
        if self.color == BASE_COLOR:
            region_color = 'blue'
        else:
            region_color = self.color
        color[self.mask] = region_color
        return color


class ModelRegion(Region):
    """
    supercharged Region with an explainable predictive model
    """

    def __init__(self,
                 X,
                 y,
                 X_test,
                 y_test,
                 customer_model,
                 rules: RuleSet | None = None,
                 mask: pd.Series | None = None,
                 color=None,
                 score=None,
                 num=-1):
        """

        Parameters
        ----------
        X: base train dataset
        y: relative target
        X_test: test dataset
        y_test: relative target
        customer_model: customer model
        rules: list of rules defining the region
        mask: mask defining the region
        color: region's color
        score: customer provided scoring method
        """
        super().__init__(X, rules, mask, color, num)
        self.y = y
        self.X_test = X_test
        self._test_mask = None
        self.y_test = y_test
        self.customer_model = customer_model
        self.interpretable_models = InterpretableModels(score)

    def to_dict(self):
        """
        transform region to dict
        Returns
        -------

        """
        dict_form = super().to_dict()
        if self.interpretable_models.selected_model is not None:
            dict_form[
                'Sub-model'] = self.interpretable_models.selected_model_str()
        dict_form["Average"] = format_number(self.y[self.mask].mean())
        return dict_form

    def select_model(self, model_name: str):
        """
        select a model between all interpretable models
        Parameters
        ----------
        model_name : model to select

        Returns
        -------

        """
        self.interpretable_models.select_model(model_name)

    def train_substitution_models(self, task_type):
        """
        train substitution models
        Returns
        -------

        """
        if self.X_test is not None:
            self.interpretable_models.get_models_performance(
                self.customer_model,
                self.X.loc[self.mask],
                self.y.loc[self.mask],
                self.X_test.loc[self.test_mask],
                self.y_test.loc[self.test_mask],
                task_type=task_type)
        else:
            self.interpretable_models.get_models_performance(
                self.customer_model,
                self.X.loc[self.mask],
                self.y.loc[self.mask],
                None,
                None,
                task_type=task_type)

    @property
    def perfs(self):
        """
        get model performance statistics
        Returns
        -------

        """
        perfs = self.interpretable_models.perfs
        if len(perfs) == 0:
            return perfs
        return perfs.sort_values('delta', ascending=True)

    @property
    def delta(self):
        """
        get performance difference between selected model and customer model
        Returns
        -------

        """
        if self.interpretable_models.selected_model:
            return self.interpretable_models.perfs.loc[
                self.interpretable_models.selected_model, 'delta']
        return 0

    def train_residuals(self, model_name: str):
        return self.y[self.mask] - self.get_model(model_name).predict(
            self.X[self.mask])

    @property
    def test_mask(self):
        """
        select testing sample from test set
        Returns
        -------

        """
        if self._test_mask is None:
            if self.rules and len(self.rules) > 0:
                self._test_mask = self.rules.get_matching_mask(self.X_test)
            else:
                if not self.mask.any() or self.mask.all():
                    return boolean_mask(self.X_test,
                                        self.mask.mean()).astype(bool)
                knn = KNeighborsClassifier().fit(self.X, self.mask)
                self._test_mask = pd.Series(knn.predict(self.X_test),
                                            index=self.X_test.index)
        return self._test_mask

    def get_model(self, model_name):
        return self.interpretable_models.models[model_name]

    def get_selected_model(self):
        if self.interpretable_models.selected_model is None:
            return None
        return self.get_model(self.interpretable_models.selected_model)

    def predict(self, X):
        mask = self.rules.get_matching_mask(X)
        model = self.get_selected_model()
        if model is not None:
            return model.predict(X[mask]).reindex(X.index)
        return pd.Series(index=X.index)

    def update_rule_set(self, rule_set: RuleSet):
        super().update_rule_set(rule_set)
        self.interpretable_models.reset()

    def update_mask(self, mask: pd.Series):
        super().update_mask(mask)
        self.interpretable_models.reset()
