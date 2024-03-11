from __future__ import annotations

import numpy as np
import pandas as pd

import os
from antakia_core.utils.utils import boolean_mask, format_number
from antakia_core.utils.variable import Variable


class Rule:
    """
    class to represent logical rule over a variable
    """

    def __init__(self, variable: Variable, min: float | None = None, includes_min: bool | None = None,
                 max: float | None = None, includes_max: bool | None = None, cat_values: list | set | None = None):
        self.variable = variable
        if cat_values is not None:
            self.cat_values = set(cat_values)
            self.min = None
            self.max = None
            self.includes_min = None
            self.includes_max = None
            self.categorical_rule = True
        else:
            self.categorical_rule = False
            self.cat_values = None
            if min is None:
                self.min = -np.inf
                self.includes_min = False
            else:
                self.min = min
                self.includes_min = includes_min if includes_min else False
            if max is None:
                self.max = np.inf
                self.includes_max = False
            else:
                self.max = max
                self.includes_max = includes_max if includes_max else False

    def __eq__(self, other):
        if self.rule_type != other.rule_type or self.variable != other.variable:
            return False
        if self.rule_type in (-1, 5):
            return True
        if self.rule_type == 0:
            return len(self.cat_values.symmetric_difference(other.cat_values)) == 0
        if self.rule_type == 4:
            return self.min == other.min
        if self.rule_type == 1:
            return self.max == other.max and self.includes_max == other.includes_max
        if self.rule_type == 2:
            return self.min == other.min and self.includes_min == other.includes_min
        return (
            self.max == other.max and self.includes_max == other.includes_max
        ) and (
            self.min == other.min and self.includes_min == other.includes_min
        )

    def __repr__(self):
        if self.is_categorical_rule:
            txt = f"{self.variable.display_name} \u2208  \u27E6"
            txt += ', '.join(self.cat_values)
            txt += "\u27E7"
            return txt
        if self.rule_type == -1:
            return f'{self.variable.display_name} - True'
        if self.rule_type == 5:
            return f'{self.variable.display_name} - False'
        if self.rule_type == 1:
            # Rule type 1
            op = '\u2264' if self.includes_max else '<'
            txt = f"{self.variable.display_name} {op} {format_number(self.max)}"
            return txt
        if self.rule_type == 2:
            # Rule type 2
            op = '\u2265' if self.includes_min else '>'
            txt = f"{self.variable.display_name} {op} {format_number(self.min)}"
            return txt
        if self.rule_type == 3:
            # Rule type 3 : the rule is of the form : variable included in [min, max] interval, or min < variable < max
            if os.environ.get("USE_INTERVALS_FOR_RULES"):
                open_bracket = '\u27E6' if self.includes_min else '['
                close_bracket = '\u27E7' if self.includes_min else ']'
                txt = f"{self.variable.display_name} \u2208 {open_bracket} {format_number(self.min)},"
                txt += f" {format_number(self.max)} {close_bracket}"  # element of
                return txt
            min_op = '\u2264' if self.includes_min else '<'
            max_op = '\u2264' if self.includes_max else '<'

            txt = f'{format_number(self.min)} {min_op} {self.variable.display_name} '
            txt += f'{max_op} {format_number(self.max)}'
            return txt
        # Rule type 4 : the rule is of the form : variable not included in [min, max] interval or variable < min and variable > max
        return f'{self.variable.display_name} = {self.min}'

    @property
    def rule_type(self):
        """
        return type of rule :
            -1: no rule
            0: categorical rule
            1: x <(=) max
            2: x >(=) min
            3: min <(=) x <(=) max
            4: x = value
            5: Falsy rule
        """
        if self.is_categorical_rule:
            if self.cat_values:
                return 0
            else:
                return 5
        if self.min == -np.inf and self.max == np.inf:
            return -1
        if self.min == -np.inf:
            return 1
        if self.max == np.inf:
            return 2
        if self.max > self.min:
            return 3
        if self.max == self.min and self.includes_max and self.includes_min:
            return 4
        return 5

    @property
    def operator_max(self):
        if self.rule_type == 4:
            return '__eq__'
        if self.includes_max:
            return '__le__'
        return '__lt__'

    @property
    def operator_min(self):
        if self.rule_type == 4:
            return '__eq__'
        if self.includes_min:
            return '__ge__'
        return '__gt__'

    @property
    def is_categorical_rule(self):
        return self.categorical_rule

    def get_matching_mask(self, X: pd.DataFrame) -> pd.Series:
        col = X.loc[:, self.variable.column_name]
        if self.rule_type == -1:
            return boolean_mask(X, True)
        if self.rule_type == 0:
            return col.isin(self.cat_values)
        if self.rule_type == 4:
            return col == self.min
        if self.rule_type == 5:
            return boolean_mask(X, False)
        return getattr(col, self.operator_max)(self.max) & getattr(col, self.operator_min)(self.min)

    def __call__(self, value: float | pd.DataFrame) -> bool | pd.Series:
        if isinstance(value, pd.DataFrame):
            return self.get_matching_mask(value)
        if self.rule_type == -1:
            return True
        if self.rule_type == 0:
            return value in self.cat_values
        if self.rule_type == 4:
            return value == self.min
        if self.rule_type == 5:
            return False
        return getattr(value, self.operator_max)(self.max) & getattr(value, self.operator_min)(self.min)

    def combine(self, rule: Rule):
        """
        builds a new combining current rule and argument rules (logical and)
        Parameters
        ----------
        rule

        Returns
        -------

        """

        if self.variable != rule.variable:
            raise ValueError('cannot combine two rules on different variables')
        # categorical rules
        if self.categorical_rule and rule.categorical_rule:
            common_values = set(self.cat_values).intersection(rule.cat_values)
            return Rule(self.variable, cat_values=common_values)
        if rule.categorical_rule:
            return rule.combine(self)
        if self.categorical_rule:
            return Rule(self.variable, cat_values=[v for v in self.cat_values if rule(v)])
        # edge cases
        if self.rule_type == 5 or rule.rule_type == 5:
            return self.copy()
        if self.rule_type == -1:
            return rule.copy()
        if rule.rule_type == -1:
            return self.copy()
        # nominal
        min_val = max(self.min, rule.min)
        if self.min > rule.min:
            include_min = self.includes_min
        elif rule.min > self.min:
            include_min = rule.includes_min
        else:
            include_min = min(self.includes_min, rule.includes_min)

        max_val = min(self.max, rule.max)
        if self.max < rule.max:
            include_max = self.includes_max
        elif rule.max < self.max:
            include_max = rule.includes_max
        else:
            include_max = min(self.includes_max, rule.includes_max)
        return Rule(self.variable, min_val, include_min, max_val, include_max)

    def __and__(self, other):
        return self.combine(other)

    def to_dict(self):
        return {
            'Variable': self.variable.display_name,
            'Unit': self.variable.unit,
            'Desc': self.variable.descr,
            'Critical': self.variable.critical,
            'Rule': self.__repr__()
        }

    def copy(self):
        return Rule(self.variable, self.min, self.includes_min, self.max, self.includes_max, self.cat_values)


class TruthyRule(Rule):
    def __init__(self, var: Variable):
        super().__init__(var)


class FalsyRule(Rule):
    def __init__(self, var: Variable):
        super().__init__(var, min=5, max=4)
