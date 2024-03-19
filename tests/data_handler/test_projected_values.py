import unittest

import numpy as np
import mock
import pandas as pd

from src.antakia_core.data_handler.projected_values import ProjectedValues, Proj
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import DummyCallable


class TestProjectedValues(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(generate_corner_dataset(10)[0])
        self.y = pd.DataFrame(generate_corner_dataset(10)[1])

    def test_init(self):
        pv = ProjectedValues(self.X, self.y)
        assert pv._projected_values == {}
        assert pv._parameters == {}

    @mock.patch('antakia_core.data_handler.projected_values.compute_projection')
    def test_set_parameters(self, cpt_proj):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)

        proj = Proj(1, 2)
        pv.compute(proj, callback.call)
        pv.set_parameters(proj, {'n_neighbors': 2})
        assert pv._parameters == {proj: {'current': {'n_neighbors': 2}, 'previous': {}}}

        pv.compute(proj, callback.call)
        pv.set_parameters(proj, {'MN_ratio': 4})
        assert pv._parameters == {proj: {'current': {'MN_ratio': 4, 'n_neighbors': 2}, 'previous': {'n_neighbors': 2}}}

        proj = Proj(2, 3)
        pv.compute(proj, callback.call)
        pv.set_parameters(proj, {'n_neighbors': 2})
    def test_get_parameters(self):
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        assert pv.get_parameters(proj) == {'current': {}, 'previous': {}}
        pv1 = ProjectedValues(self.X, self.y)
        proj = Proj(2, 3)
        pv1.build_default_parameters(proj)
        assert pv1.get_parameters(proj) == {
            'current': {'min_dist': 0.1, 'n_neighbors': 15},
            'previous': {'min_dist': 0.1, 'n_neighbors': 15}
        }

    def test_build_default_parameters(self):
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        pv.build_default_parameters(proj)
        assert pv._parameters == {proj: {'current': {}, 'previous': {}}}

        pv1 = ProjectedValues(self.X, self.y)
        proj = Proj(2, 3)
        pv1.build_default_parameters(proj)
        assert pv1._parameters == {
            Proj(reduction_method=2, dimension=3): {
                'current': {'min_dist': 0.1, 'n_neighbors': 15},
                'previous': {'min_dist': 0.1, 'n_neighbors': 15}
            }
        }

    @mock.patch('antakia_core.data_handler.projected_values.compute_projection')
    def test_get_projection(self, cpt_proj):
        X_red = pd.DataFrame([[4, 7, 10],
                              [5, 8, 11],
                              [6, 9, 12]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c'])
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        pv.get_projection(proj)

        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        pv._projected_values = {proj: X_red}
        np.testing.assert_array_equal(pv.get_projection(proj), X_red)

        cpt_proj.return_value = X_red
        pv = ProjectedValues(self.X, self.y)
        pv._projected_values = {proj: X_red}
        np.testing.assert_array_equal(pv.get_projection(proj), X_red)

    @mock.patch('antakia_core.data_handler.projected_values.compute_projection')
    def test_is_present(self, cpt_proj):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        assert not pv.is_present(proj)

        pv.compute(proj, callback.call)
        assert pv.is_present(proj)

    @mock.patch('antakia_core.data_handler.projected_values.compute_projection')
    def test_compute(self, cpt_proj):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)
        X_red = pd.DataFrame([[4, 7, 10],
                              [5, 8, 11],
                              [6, 9, 12]],
                             index=[1, 2, 3],
                             columns=['a', 'b', 'c'])

        cpt_proj.return_value = X_red
        proj = Proj(1, 2)

        pv.compute(proj, callback.call)
        assert pv._projected_values[proj].shape == (self.X.shape[0], proj.dimension)