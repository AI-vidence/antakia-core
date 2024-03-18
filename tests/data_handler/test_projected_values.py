import numpy as np
import mock
import pandas as pd

from antakia_core.data_handler.projected_values import ProjectedValues, Proj
from tests.utils_fct import generate_df_series, DummyCallable


def test_init():
    X, y = generate_df_series()
    pv = ProjectedValues(X, y)
    assert pv.X.equals(X)
    assert pv.y.equals(y)
    assert pv._projected_values == {}
    assert pv._parameters == {}


@mock.patch('antakia_core.data_handler.projected_values.compute_projection')
def test_set_parameters(cpt_proj):
    X, y = generate_df_series()
    proj = Proj(1, 2)
    callback = DummyCallable()
    pv = ProjectedValues(X, y)
    pv.compute(proj, callback)
    pv.set_parameters(proj, {'n_neighbors': 2})
    assert pv._parameters == {
        proj: {
            'current': {
                'n_neighbors': 2
            },
            'previous': {}
        }
    }
    pv.compute(proj, callback)
    pv.set_parameters(proj, {'MN_ratio': 4})
    assert pv._parameters == {
        proj: {
            'current': {
                'MN_ratio': 4,
                'n_neighbors': 2
            },
            'previous': {
                'n_neighbors': 2
            }
        }
    }


def test_get_parameters():
    X, y = generate_df_series()
    pv = ProjectedValues(X, y)
    proj = Proj(1, 2)
    assert pv.get_parameters(proj) == {'current': {}, 'previous': {}}
    pv1 = ProjectedValues(X, y)
    proj = Proj(2, 3)
    pv1.build_default_parameters(proj)
    assert pv1.get_parameters(proj) == {
        'current': {
            'min_dist': 0.1,
            'n_neighbors': 15
        },
        'previous': {
            'min_dist': 0.1,
            'n_neighbors': 15
        }
    }


def test_build_default_parameters():
    X, y = generate_df_series()
    pv = ProjectedValues(X, y)
    proj = Proj(1, 2)
    pv.build_default_parameters(proj)
    assert pv._parameters == {proj: {'current': {}, 'previous': {}}}

    pv1 = ProjectedValues(X, y)
    proj = Proj(2, 3)
    pv1.build_default_parameters(proj)
    assert pv1._parameters == {
        Proj(reduction_method=2, dimension=3): {
            'current': {
                'min_dist': 0.1,
                'n_neighbors': 15
            },
            'previous': {
                'min_dist': 0.1,
                'n_neighbors': 15
            }
        }
    }


@mock.patch('antakia_core.data_handler.projected_values.compute_projection')
def test_get_projection(cpt_proj):
    X_red = pd.DataFrame([[4, 7, 10], [5, 8, 11], [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])
    X, y = generate_df_series()
    pv = ProjectedValues(X, y)
    proj = Proj(1, 2)
    pv._projected_values = {proj: X_red}
    np.testing.assert_array_equal(pv.get_projection(proj), X_red)

    cpt_proj.return_value = X_red
    pv1 = ProjectedValues(X, y)
    pv1._projected_values = {proj: X_red}
    np.testing.assert_array_equal(pv1.get_projection(proj), X_red)


@mock.patch('antakia_core.data_handler.projected_values.compute_projection')
def test_is_present(cpt_proj):
    X, y = generate_df_series()
    callback = DummyCallable()
    pv = ProjectedValues(X, y)
    proj = Proj(1, 2)
    assert not pv.is_present(proj)

    pv.compute(proj, callback)
    assert pv.is_present(proj)


@mock.patch('antakia_core.data_handler.projected_values.compute_projection')
def test_compute(cpt_proj):
    X, y = generate_df_series()
    callback = DummyCallable()
    pv = ProjectedValues(X, y)
    X_red = pd.DataFrame([[4, 7, 10], [5, 8, 11], [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])

    cpt_proj.return_value = X_red
    proj = Proj(1, 2)

    pv.compute(proj, callback)
    assert pv._projected_values[proj].equals(X_red)
