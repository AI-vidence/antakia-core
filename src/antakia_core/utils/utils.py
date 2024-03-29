"""
Utils module for the antakia package.
"""
import math
import time
from enum import EnumMeta, Enum
from numbers import Number

import numpy as np
from functools import wraps

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype


def in_index(indexes: list, X: pd.DataFrame) -> bool:
    """
    Checks if a list of indexes is in the index of a DataFrame
    """
    try:
        _ = X.loc[indexes]
        return True
    except KeyError:
        return False


def rows_to_mask(X: pd.DataFrame, rows_list: list[int]) -> pd.Series:
    """
    Converts DataFrame row numbers to Index numbers
    """
    mask = boolean_mask(X, False)
    mask.iloc[rows_list] = 1
    return mask.astype(bool)


def indexes_to_rows(X: pd.DataFrame, indexes_list: list) -> list:
    """
    Converts DataFrame Index numbers to row numbers
    """
    index = pd.Series(np.arange(len(X)), index=X.index)
    return index.loc[indexes_list].tolist()


def mask_to_rows(mask: pd.Series) -> list:
    """
    converts a mask to row indices (i.e. iloc)
    """
    return mask_to_index(mask.reset_index(drop=True))


def mask_to_index(mask: pd.Series) -> list:
    """
    converts a mask to indices (i.e. loc)
    """
    return mask[mask].index.tolist()


def boolean_mask(X: pd.DataFrame, value: bool = True):
    """
    builds a constant series indexed on X with value as value
    """
    return pd.Series([value] * len(X), index=X.index)


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def debug(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result

    return wrapper


def compute_step(min, max):
    """
    compute rounded min, max, and step values
    """
    step = (max - min) / 100
    round_value = round(math.log(step / 2) / math.log(10)) - 1
    step = np.round(step, -round_value)
    min_ = np.round(min, -round_value)
    max_ = int((max - min_) / step + 1) * step + min
    return min_, max_, step


def get_mask_comparison_color(rules_mask, selection_mask):
    """
    compute colors for comparison between two masks
    """
    colors_info = {
        'matched': 'blue',
        'error type 1': 'orange',
        'error type 2': 'red',
        'other data': BASE_COLOR
    }
    color = pd.Series(index=selection_mask.index, dtype=str)
    color[selection_mask & rules_mask] = colors_info['matched']
    color[~selection_mask & rules_mask] = colors_info['error type 1']
    color[selection_mask & ~rules_mask] = colors_info['error type 2']
    color[~selection_mask & ~rules_mask] = colors_info['other data']
    return color, colors_info


# First color can't be blue, reserved for the rules - grey is reserved to background
colors = [
    "red", "blue", "green", "yellow", "orange", "pink", "brown", "cyan",
    "black"
]
BASE_COLOR = 'grey'


class MetaEnum(EnumMeta):

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


def format_data(x):
    if isinstance(x, str):
        return x[:15]
    if isinstance(x, Number):
        return format_number(x)
    if isinstance(x, pd.Series):
        if is_bool_dtype(x.dtype) or is_integer_dtype(x.dtype):
            return x
        return x.apply(format_data)
    return x


def format_number(value: Number) -> str:

    def format_str(value):
        if abs(value) < 1e-2 or abs(value) > 1e4:
            return '.2e'
        return '.2f'

    if value == 0:
        return str(value)
    elif pd.isna(value):  # type:ignore
        return 'NaN'
    return format(value, format_str(value))


ProblemCategory = Enum('ProblemCategory', [
    'classification_with_proba', 'classification_proba', 'classification',
    'classification_label_only', 'regression', 'auto'
])
