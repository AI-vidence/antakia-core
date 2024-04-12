from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from antakia_core.utils.variable import Variable, DataVariables


class TestVariable(TestCase):
    def setUp(self) -> None:
        self.var1 = dict(col_index=0,
                         column_name='var1',
                         display_name='display_var1',
                         formatter='seconds',
                         descr='description',
                         critical=True,
                         continuous=True,
                         lat=True,
                         lon=True,
                         used_for_prediction=False)
        self.short_v2 = dict(column_name='var2', col_index=0)
        self.variables_df = pd.DataFrame(
            {
                'col_index': [0, 1, 2, 3, 4, 5, 6, 7],
                'descr': [
                    'Median income', 'House age', 'Average nb rooms',
                    'Average nb bedrooms', 'Population', 'Average occupancy',
                    'Latitude', 'Longitude'
                ],
                'col_type': [
                    'continuous', 'continuous', 'continuous', 'discrete', 'continuous', 'continuous',
                    'continuous', 'continuous'
                ],
                'unit': [
                    'k$', 'years', 'rooms', 'rooms', 'people', 'ratio', 'degrees',
                    'degrees'
                ],
                'critical':
                    [True, False, False, False, False, False, False, False],
                'lat': [False, False, False, False, False, False, True, False],
                'lon': [False, False, False, False, False, False, False, True]
            },
            index=[
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
                'AveOccup', 'Latitude', 'Longitude'
            ])
        self.list_var = [{
            'col_index': 0,
            "column_name": 'a',
            'col_type': 'discrete'
        }, {
            'col_index': 1,
            "column_name": 'b',
            'col_type': 'discrete'
        }, {
            'col_index': 2,
            "column_name": 'c',
            'col_type': 'discrete'
        }]
        self.X1 = pd.DataFrame()
        self.X2 = pd.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9], "c": [10, 11, 12]})
        self.X3 = pd.DataFrame({"lat": [5]})
        self.X4 = pd.DataFrame({"long": [5]})
        self.X = pd.DataFrame(np.random.random((20, 8)), columns=[
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
            'AveOccup', 'Latitude', 'Longitude'
        ])

    def test_init(self):
        var = Variable(**self.var1)
        assert var.col_index == 0
        assert var.column_name == 'var1'
        assert var.col_type == 'auto'
        assert var.formatter(1) == '1seconds'
        assert var.descr == 'description'
        assert var.critical
        assert not var.used_for_prediction
        assert var.lat
        assert var.lon

        var = Variable(**self.short_v2)

        assert var.col_index == 0
        assert var.column_name == 'var2'
        assert var.display_name == 'var2'
        assert var.col_type == 'auto'
        assert not var.critical
        assert var.used_for_prediction
        assert not var.lat
        assert not var.lon

    def test_build_variable_none(self):
        assert DataVariables.build_variables(self.X1) == DataVariables([])
        assert DataVariables.build_variables(self.X2) == DataVariables([
            Variable(0, 'a', col_type='continuous', continuous=True),
            Variable(1, 'b', col_type='continuous', continuous=True),
            Variable(2, 'c', col_type='continuous', continuous=True)
        ])
        assert DataVariables.build_variables(self.X3) == DataVariables(
            [Variable(0, 'lat', col_type='continuous', lat=True, continuous=True)])

        assert DataVariables.build_variables(self.X4) == DataVariables(
            [Variable(0, 'long', col_type='continuous', lon=True, continuous=True)])

    def test_import_variable_df(self):
        assert DataVariables.build_variables(self.X, self.variables_df) == DataVariables([
            Variable(0,
                     'MedInc',
                     col_type='continuous',
                     descr='Median income',
                     unit='k$',
                     critical=True),
            Variable(1, 'HouseAge', col_type='continuous', descr='House age', unit='years'),
            Variable(2,
                     'AveRooms',
                     col_type='continuous',
                     descr='Average nb rooms',
                     unit='rooms'),
            Variable(3,
                     'AveBedrms',
                     col_type='discrete',
                     descr='Average nb bedrooms',
                     unit='rooms'),
            Variable(4, 'Population', col_type='continuous', descr='Population', unit='people'),
            Variable(5,
                     'AveOccup',
                     col_type='continuous',
                     descr='Average occupancy',
                     unit='ratio'),
            Variable(6,
                     'Latitude',
                     col_type='continuous',
                     descr='Latitude',
                     unit='degrees',
                     lat=True),
            Variable(7,
                     'Longitude',
                     col_type='continuous',
                     descr='Longitude',
                     unit='degrees',
                     lon=True)
        ])

        variables_df1 = pd.DataFrame({
            'col_index': [0],
            'col_type': ['continuous']
        }, index=['MedInc'])

        assert DataVariables.build_variables(self.X[['MedInc']], variables_df1) == DataVariables(
            [Variable(0, 'MedInc', col_type='continuous')])

        variables_df2 = pd.DataFrame({
            'colonne': [0],
            'col_type': ['continuous']
        },
            index=['MedInc'])

        assert DataVariables.build_variables(self.X[['MedInc']], variables_df2) == DataVariables(
            [Variable(0, 'MedInc', col_type='continuous')])

        with pytest.raises(KeyError):
            DataVariables.build_variables(self.X[['MedInc']],
                                          variables_df1.drop('column_name', axis=1).reset_index(drop=True))
        DataVariables.build_variables(self.X[['MedInc']], variables_df1.drop('col_type', axis=1))


def test_import_variable_list():
    list_var = [{
        'col_index': 0,
        "column_name": 'a',
        'col_type': 'discrete'
    }, {
        'col_index': 1,
        "column_name": 'b',
        'col_type': 'discrete'
    }, {
        'col_index': 2,
        "column_name": 'c',
        'col_type': 'discrete'
    }]

    assert DataVariables.import_variable_list(list_var) == DataVariables([
        Variable(0, 'a', col_type='discrete'),
        Variable(1, 'b', col_type='discrete'),
        Variable(2, 'c', col_type='discrete')
    ])

    list_var1 = [{
        'colonne_index': 0,
        "column_name": 'a',
        'col_type_de_variable': 'discrete'
    }]

    with pytest.raises(ValueError):
        DataVariables.import_variable_list(list_var1)


def test_repr():
    var1 = Variable(0, 'var1', col_type='int')
    assert repr(var1) == "var1, col#:0, type:int"

    var2 = Variable(0,
                    'var2',
                    col_type='int',
                    unit='seconds',
                    descr='description',
                    critical=True,
                    continuous=False,
                    lat=True,
                    lon=True)
    assert repr(
        var2
    ) == "var2, col#:0, type:int, descr:description, unit:seconds, critical, is lat, is lon"


def test_str_dv():
    dv = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert str(dv) == (
        '0) MedInc, col#:0, type:continuous, descr:Median income, unit:k$, critical\n'
        '6) Latitude, col#:6, type:continuous, descr:Latitude, unit:degrees, is lat\n'
        '7) Longitude, col#:7, type:continuous, descr:Longitude, unit:degrees, is lon\n'
    )


def test_sym_list():
    dv = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
        Variable(2,
                 'AveRooms',
                 col_type='continuous',
                 descr='Average nb rooms',
                 unit='rooms'),
        Variable(3,
                 'AveBedrms',
                 col_type='continuous',
                 descr='Average nb bedrooms',
                 unit='rooms'),
        Variable(4, 'Population', 'int', descr='Population', unit='people'),
        Variable(5,
                 'AveOccup',
                 col_type='continuous',
                 descr='Average occupancy',
                 unit='ratio'),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert dv.columns_list() == [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude'
    ]


def test_get_var():
    dv = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
        Variable(2,
                 'AveRooms',
                 col_type='continuous',
                 descr='Average nb rooms',
                 unit='rooms'),
        Variable(3,
                 'AveBedrms',
                 col_type='continuous',
                 descr='Average nb bedrooms',
                 unit='rooms'),
        Variable(4, 'Population', 'int', descr='Population', unit='people'),
        Variable(5,
                 'AveOccup',
                 col_type='continuous',
                 descr='Average occupancy',
                 unit='ratio'),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert dv.get_var('MedInc') == Variable(0,
                                            'MedInc',
                                            col_type='continuous',
                                            descr='Median income',
                                            unit='k$',
                                            critical=True)


def test_len_dv():
    dv = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(1, 'HouseAge', col_type='int', descr='House age', unit='years'),
        Variable(2,
                 'AveRooms',
                 col_type='continuous',
                 descr='Average nb rooms',
                 unit='rooms'),
        Variable(3,
                 'AveBedrms',
                 col_type='continuous',
                 descr='Average nb bedrooms',
                 unit='rooms'),
        Variable(4, 'Population', col_type='int', descr='Population', unit='people'),
        Variable(5,
                 'AveOccup',
                 col_type='continuous',
                 descr='Average occupancy',
                 unit='ratio'),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert len(dv) == 8


def test_eq_dv():
    dv = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])
    dv1 = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(3,
                 'AveBedrms',
                 col_type='continuous',
                 descr='Average nb bedrooms',
                 unit='rooms'),
        Variable(4, 'Population', 'int', descr='Population', unit='people'),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert not dv == dv1

    dv2 = DataVariables([
        Variable(0,
                 'MedInc',
                 col_type='continuous',
                 descr='Median income',
                 unit='k$',
                 critical=True),
        Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
        Variable(3,
                 'AveBedrms',
                 col_type='continuous',
                 descr='Average nb bedrooms',
                 unit='rooms'),
        Variable(4, 'Population', 'int', descr='Population', unit='people'),
        Variable(6,
                 'Latitude',
                 col_type='continuous',
                 descr='Latitude',
                 unit='degrees',
                 lat=True),
        Variable(7,
                 'Longitude',
                 col_type='continuous',
                 descr='Longitude',
                 unit='degrees',
                 lon=True)
    ])

    assert not dv1 == dv2
