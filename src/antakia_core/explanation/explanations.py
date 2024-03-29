from __future__ import annotations
from typing import Callable

import lime
import numpy as np
import pandas as pd
import shap

from antakia_core.explanation.explanation_method import ExplanationMethod
from antakia_core.utils.utils import ProblemCategory

# ===========================================================
#              Explanations implementations
# ===========================================================


class SHAPExplanation(ExplanationMethod):
    """
    SHAP computation class.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 model,
                 task_type,
                 progress_updated: Callable | None = None):
        super().__init__(ExplanationMethod.SHAP, X, model, task_type,
                         progress_updated)

    @property
    def link(self):
        if self.task_type == ProblemCategory.regression:
            return "identity"
        return "logit"

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)
        try:
            explainer = shap.TreeExplainer(self.model)
        except:
            explainer = shap.KernelExplainer(self.model.predict,
                                             self.X.sample(
                                                 min(200, len(self.X))),
                                             link=self.link)
        chunck_size = int(max(200, len(self.X) / 100))
        shap_val_list = []
        for i in range(0, len(self.X), chunck_size):
            explanations = explainer.shap_values(self.X.iloc[i:i +
                                                             chunck_size])
            if isinstance(explanations, list):
                # classification, use only class 1 probabilities
                explanations = explanations[-1]
            shap_val_list.append(
                pd.DataFrame(explanations,
                             columns=self.X.columns,
                             index=self.X.index[i:i + chunck_size]))
            self.publish_progress(int(100 * i / len(self.X)))
        shap_values = pd.concat(shap_val_list)
        self.publish_progress(100)
        return shap_values


class LIMExplanation(ExplanationMethod):
    """
    LIME computation class.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 model,
                 task_type,
                 progress_updated: Callable | None = None):
        super().__init__(ExplanationMethod.LIME, X, model, task_type,
                         progress_updated)

    @property
    def mode(self):
        if self.task_type == ProblemCategory.regression:
            return 'regression'
        else:
            return 'classification'

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X.sample(min(len(self.X), 500)).values,
            feature_names=self.X.columns,
            verbose=False,
            mode=self.mode,
            discretize_continuous=False)

        values_lime = pd.DataFrame(np.zeros(self.X.shape),
                                   index=self.X.index,
                                   columns=self.X.columns)
        progress = 0.
        if self.mode == 'regression':
            predict_fct = self.model.predict
            i = 0
        else:
            i = 1
            if hasattr(self.model, 'predict_proba'):
                predict_fct = self.model.predict_proba
            else:
                predict_fct = self.model.predict
        for index, row in self.X.iterrows():
            exp = explainer.explain_instance(row.values, predict_fct)

            values_lime.loc[index] = pd.Series(  # type:ignore
                exp.local_exp[i],
                index=explainer.feature_names).str[1][  # type:ignore
                    values_lime.columns].values  # type:ignore
            progress += 100. / len(self.X)
            self.publish_progress(int(progress))
        self.publish_progress(100)
        return values_lime


def compute_explanations(X: pd.DataFrame, model, explanation_method: int,
                         task_type,
                         progress_callback: Callable | None) -> pd.DataFrame:
    """ Generic method to compute explanations, SHAP or LIME
    """
    if explanation_method == ExplanationMethod.SHAP:
        return SHAPExplanation(X, model, task_type,
                               progress_callback).compute()
    elif explanation_method == ExplanationMethod.LIME:
        return LIMExplanation(X, model, task_type, progress_callback).compute()
    else:
        raise ValueError(
            f"This explanation method {explanation_method} is not valid!")
