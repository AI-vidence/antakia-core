from __future__ import annotations
import pandas as pd

from antakia_core.data_handler.rule import Rule
from antakia_core.utils.utils import boolean_mask, mask_to_index
from antakia_core.utils.variable import Variable, DataVariables


class RuleSet:
    """
    set of rules
    """

    def __init__(self, rules: list[Rule] | RuleSet | None = None):

        if isinstance(rules, RuleSet):
            rules = rules.rules.values()
        self.rules = {}
        if rules:
            for rule in rules:
                self.rules[rule.variable] = rule

    def add(self, value: Rule):
        """
        add a new rule
        combines it with the existing rule is variable already used
        Parameters
        ----------
        value

        Returns
        -------

        """
        if value.variable in self.rules:
            self.rules[value.variable] &= value
        self.rules[value.variable] = value

    def replace(self, value: Rule):
        """
        add or replace (if variable is already used) a new rule
        Parameters
        ----------
        value

        Returns
        -------

        """
        self.rules[value.variable] = value

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        if not self.rules:
            return ""
        return " and ".join([rule.__repr__() for rule in self.rules.values()])

    def to_dict(self):
        if not self.rules:
            return []
        return [rule.to_dict() for rule in self.rules.values()]

    def copy(self):
        return RuleSet(self.rules.values())

    def get_matching_mask(self, X: pd.DataFrame) -> pd.Series:
        """
        get the mask of samples validating the rule
        Parameters
        ----------
        X: dataset to get the mask from

        Returns
        -------

        """
        res = boolean_mask(X, True)
        if self.rules is not None:
            for rule in self.rules.values():
                res &= rule.get_matching_mask(X)
        return res

    def get_all_masks(self, X: pd.DataFrame) -> list[pd.Series]:
        """
        returns the list of rules masks on X
        Parameters
        ----------
        X

        Returns
        -------

        """
        masks = []
        if self.rules is not None:
            for rule in self.rules.values():
                masks.append(rule.get_matching_mask(X))
        return masks

    def get_matching_indexes(self, X):
        """
        get the list indexes of X validating the rule
        Parameters
        ----------
        X

        Returns
        -------

        """
        res = self.get_matching_mask(X)
        return mask_to_index(res)

    @classmethod
    def sk_rules_to_rule_set(cls, skrules, variables: DataVariables):
        """
        transform skope rules to a RuleSet
        Parameters
        ----------
        skrules
        variables

        Returns
        -------

        """
        rules_info = skrules[0]
        precision, recall, __ = rules_info[1]
        f1 = precision * recall * 2 / (precision + recall)
        score_dict = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
        rule_strings = rules_info[0].split(" and ")

        rule_list = RuleSet()
        for rule in rule_strings:
            rule_parts = rule.split(' ')

            variable = variables.get_var(rule_parts[0])
            if len(rule_parts) != 3:
                raise ValueError('Rule not recognized')
            includes = '=' in rule_parts[1]
            value = float(rule_parts[2])
            if '<' in rule_parts[1]:
                temp_rule = Rule(variable, max=value, includes_max=includes)
            else:
                temp_rule = Rule(variable, min=value, includes_min=includes)
            rule_list.add(temp_rule)

        return rule_list, score_dict

    def get_rule(self, var: Variable) -> Rule | None:
        """
        find a rule on the variable
        Parameters
        ----------
        var

        Returns
        -------

        """
        return self.rules.get(var)
