from typing import List, Callable, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype


def preprocess_formatter(formatter: str | Callable[[Any], str] | None):
    if formatter is None:
        return lambda x: str(x)
    if isinstance(formatter, str):
        # TODO : manage decimals
        return lambda x: f'{x}{formatter}'
    return formatter


class VariableError(Exception):
    pass


class Variable:
    """
    Describes each X column or Y value

    col_index : int
        The index of the column in the dataframe/array
    column_name : str
        column_name in dataframe
    display_name : str
        How it should be displayed in the GUI
    descr : str
        A description of the variable
    col_type : str
        The col_type of the variable
    formatter : str|Callable[[Any], str]
        either the unit of the column or a callable to format columns values
    critical : bool
        Whether the variable is critical or not
    used_for_prediction : bool

    continuous : bool
    lat : bool
    lon : bool
    """
    UNKNOWN = 'unknown'
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    CATEGORICAL = 'categorical'
    BOOLEAN = 'boolean'
    DUMMY = 'dummy'
    TIMESERIE = 'timeserie'
    AVAILABLE_TYPES = [
        UNKNOWN, CONTINUOUS, DISCRETE, CATEGORICAL, BOOLEAN, DUMMY, TIMESERIE
    ]

    def __init__(
            self,
            col_index: int,
            column_name: str,
            display_name: str | None = None,
            descr: str | None = None,
            col_type: str = 'auto',
            formatter: str | Callable[[Any], str] | None = None,
            unit: str | None = None,
            critical: bool = False,
            used_for_prediction: bool = True,
            lat: bool = False,
            lon: bool = False,
            **kwargs  # to ignore unknown args in building object
    ):
        self.col_index = col_index
        self.column_name = column_name
        self._display_name = display_name
        self.col_type = col_type
        if formatter is None:
            self.formatter = preprocess_formatter(unit)
        else:
            self.formatter = preprocess_formatter(formatter)
        self.descr = descr
        self.critical = critical
        self.used_for_prediction = used_for_prediction
        self.lat = lat
        self.lon = lon
        self.main_feature = False
        self.dummy_group = None

    def update(self, **kwargs):
        for k, v in kwargs:
            if k == 'formatter':
                v = preprocess_formatter(v)
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def display_name(self):
        if self._display_name is None:
            return self.column_name.replace('_', ' ')
        return self._display_name

    @classmethod
    def get_type(cls, serie: pd.Series) -> str:
        if is_datetime64_dtype(serie.dtype):
            return cls.TIMESERIE
        if is_numeric_dtype(serie.dtype):
            unique = serie.unique()
            if len(unique) <= 2 and set(unique).issubset({True, False}):
                return cls.BOOLEAN
            if len(unique) < len(serie) / 10:
                return cls.DISCRETE
            return cls.CONTINUOUS
        return cls.UNKNOWN

    def __repr__(self):
        """
        Displays the variable as a string
        """
        text = f"{self.display_name}, col#:{self.col_index}, type:{self.col_type}"
        if self.descr is not None:
            text += f", descr:{self.descr}"
        if self.formatter is not None and self.formatter('') != '':
            text += f", unit:{self.formatter('')}"
        if self.critical:
            text += ", critical"
        if not self.used_for_prediction:
            text += ", ignored by model"
        if self.lat:
            text += ", is lat"
        if self.lon:
            text += ", is lon"
        return text

    def __eq__(self, other):
        return (self.col_index == other.col_index
                and self.column_name == other.column_name
                and self.display_name == other.display_name
                and self.used_for_prediction == other.used_for_prediction
                and self.col_type == other.col_type
                and self.formatter('') == other.formatter('')
                and self.descr == other.descr
                and self.critical == other.critical and self.lat == other.lat
                and self.lon == other.lon)

    def __hash__(self):
        return hash(self.column_name)

    @classmethod
    def guess_variable_from_serie(cls,
                                  x: pd.Series,
                                  index,
                                  column_name=None) -> 'Variable':
        """
        guess Variables from the Dataframe values
        Returns a DataVariable object, with one Variable for each column in X.
        """
        if column_name is None:
            column_name = x.name
        col_type = Variable.get_type(x)
        var = Variable(index,
                       column_name,
                       col_type=col_type,
                       lat=column_name.lower() in ["latitude", "lat"],
                       lon=column_name.lower() in ["longitude", "long", "lon"])
        return var

    @classmethod
    def from_dict(cls, var_dict: dict, index=None):
        var_dict = {**var_dict}
        if index is not None:
            var_dict['col_index'] = index
        return Variable(**var_dict)

    @classmethod
    def build_variable(cls,
                       var_dict: dict,
                       x: pd.Series,
                       index,
                       column_name=None):
        local_var_dict = {**var_dict}
        if 'col_index' not in var_dict:
            local_var_dict['col_index'] = index
        if column_name is None:
            column_name = x.name
        if 'column_name' not in var_dict:
            local_var_dict['column_name'] = column_name
        if 'col_type' not in var_dict or var_dict['col_type'] == 'auto':
            local_var_dict['col_type'] = Variable.get_type(x)
        local_var_dict['lat'] = column_name.lower() in ["latitude", "lat"]
        local_var_dict['lon'] = column_name.lower() in [
            "longitude", "long", "lon"
        ]
        var = Variable(**local_var_dict)
        return var


class DataVariables:
    """
    collection of Variables
    """

    def __init__(self, variables: List[Variable]):
        self.variables = {var.column_name: var for var in variables}

    def __str__(self):
        text = ""
        for var in self.variables.values():
            text += str(var.col_index) + ") " + str(var) + "\n"
        return text

    def columns_list(self):
        """
        get column_name list
        """
        return list(self.variables.keys())

    def training_columns_list(self):
        """
        get column_name list
        """
        return list(
            map(
                lambda x: x.column_name,
                filter(lambda x: x.used_for_prediction,
                       self.variables.values())))

    def get_var(self, column_name: str):
        """
        get variable by column_name
        """
        return self.variables.get(column_name)

    def __len__(self):
        return len(self.variables)

    def __eq__(self, other):
        for i in self.variables.values():
            if i not in other.variables.values():
                return False
        for j in other.variables.values():
            if j not in self.variables.values():
                return False
        return True

    def set_main_variables(self, main_var_names):
        for var in main_var_names:
            self.get_var(var).main_feature = True

    @classmethod
    def _dataframe_to_list(cls,
                           variable_description: pd.DataFrame) -> list[dict]:
        """
        Builds The DataVariable object from a descriptive DataFrame
        Returns a DataVariable object, with one Variable for each column in X.
        """

        if "col_index" not in variable_description.columns:
            variable_description['col_index'] = np.arange(
                len(variable_description))
        if 'column_name' not in variable_description.columns:
            variable_description['column_name'] = variable_description.index
            if is_numeric_dtype(variable_description['column_name']):
                raise KeyError(
                    'column_name (index) column is mandatory and should be string'
                )
        return variable_description.to_dict(orient='records')

    @staticmethod
    def import_variable_list(var_list: list) -> 'DataVariables':
        """
        Builds The DataVariable object from alist of dict
        Returns a DataVariable object, with one Variable for each column in X.

        """
        variables = []
        for i in range(len(var_list)):
            if isinstance(var_list[i], dict):
                item = var_list[i]
                if "col_index" in item and "column_name" in item and "col_type" in item:
                    var = Variable(**item)
                    variables.append(var)
                else:
                    raise ValueError(
                        "Variable must a list of {key:value} with mandatory keys : [col_index, column_name, col_type] and optional keys : [unit, descr, critical, continuous, lat, lon]"
                    )
        return DataVariables(variables)

    @staticmethod
    def check_variables(X, var_list):
        checked_cols = []
        for v in var_list:
            if 'column_name' in v:
                if v['column_name'] not in X.columns:
                    raise VariableError(
                        f'column name {v["column_name"]} must be in dataframe, use display name instead'
                    )
                if 'col_index' in v:
                    if X.iloc[:, v['col_index']].name != v['column_name']:
                        raise VariableError(
                            f'column index {v["col_index"]} and column name {v["column_name"]} must match'
                        )
                if v['column_name'] in checked_cols:
                    raise VariableError(
                        f'Duplicate column {v["column_name"]} at index {v["col_index"]}'
                    )
                checked_cols.append(v['column_name'])
            else:
                if 'col_index' not in v:
                    raise VariableError(
                        f'you must provide either column_name or col_index for {v}'
                    )
                column_name = X.iloc[:, v['col_index']].name
                if v['column_name'] in checked_cols:
                    raise VariableError(
                        f'Duplicate column {v["column_name"]} at index {v["col_index"]}'
                    )
                checked_cols.append(column_name)
            if 'col_type' in v:
                if v['col_type'] not in Variable.AVAILABLE_TYPES:
                    raise VariableError(
                        f'col_type {v["col_type"]} should be one of {Variable.AVAILABLE_TYPES}'
                    )

    @classmethod
    def build_variables(
            cls,
            X: pd.DataFrame,
            var_list: list[dict] | pd.DataFrame | None = None
    ) -> 'DataVariables':
        if isinstance(var_list, pd.DataFrame):
            var_list = cls._dataframe_to_list(var_list)
        if var_list is None:
            var_list = []
        cls.check_variables(X, var_list)
        var_index_dict = {}
        var_name_dict = {}
        for v in var_list:
            if var_list and 'column_name' in var_list[0]:
                var_name_dict[v['column_name']] = v
            elif 'col_index' in v:
                var_index_dict[v['col_index']] = v
        variables = []
        for i, col in enumerate(X.columns):
            if col in var_name_dict:
                var = Variable.build_variable(var_name_dict[col], X[col], i,
                                              col)
            elif i in var_index_dict:
                var = Variable.build_variable(var_index_dict[i], X[col], i,
                                              col)
            else:
                var = Variable.build_variable({}, X[col], i, col)
            variables.append(var)
        return DataVariables(variables)

    @staticmethod
    def guess_dummies(X: pd.DataFrame):
        # TODO
        pass
