import warnings

import pandas as pd
from skope_rules_temp import SkopeRules

from antakia_core.utils.variable import Variable, DataVariables
from antakia_core.data_handler.rules import Rule, RuleSet


def skope_rules(df_mask: pd.Series,
                base_space_df: pd.DataFrame,
                variables: DataVariables | None = None,
                precision: float = 0.7,
                recall: float = 0.7,
                random_state=42) -> tuple[RuleSet, dict[str, float]]:
    """
    variables : list of Variables of the app
    df_indexes : list of (DataFrame) indexes for the points selected in the GUI
    base_space_df : the dataframe on which the rules will be computed / extracted. May be VS or ES values
    precision for SKR binary classifer : defaults to 0.7
    recall for SKR binary classifer : defaults to 0.7
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We convert df_indexes in row_indexes
        y_train = df_mask.astype(int)
        if variables is None:
            variables = DataVariables.guess_variables(base_space_df)

        sk_classifier = SkopeRules(
            feature_names=variables.columns_list(),
            random_state=random_state,
            n_estimators=5,
            recall_min=recall,
            precision_min=precision,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )

        sk_classifier.fit(base_space_df, y_train)

    if sk_classifier.rules_ != []:
        rules_list, score_dict = RuleSet.sk_rules_to_rule_set(
            sk_classifier.rules_, variables)
        return rules_list, score_dict

    else:
        return RuleSet(), {}
