from unittest import TestCase

import numpy as np
import mock
import pandas as pd

from antakia_core.data_handler.projected_values import ProjectedValues, Proj
from tests.dummy_datasets import generate_corner_dataset


class TestProjectedValues(TestCase):

    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.X = pd.DataFrame(self.X)
        self.y = pd.DataFrame(self.y)

    def test_init(self):
        pv = ProjectedValues(self.X, self.y)
        assert pv.X.equals(self.X)
        assert pv.y.equals(self.y)
        assert pv._projected_values == {}
        assert pv._parameters == {}

    def test_set_parameters(self):
        proj = Proj(1, 2)
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)
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

        # pv1 = ProjectedValues(self.X, self.y)
        # pv1.set_parameters(proj, {'n_neighbors': 2})
        # trouver un test avec self._parameters.get(projection) is None

    def test_get_parameters(self):
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        assert pv.get_parameters(proj) == {'current': {}, 'previous': {}}
        pv1 = ProjectedValues(self.X, self.y)
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

    def test_get_projection(self):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)

        #get a pv that is already computed
        proj = Proj(1, 2)
        pv.compute(proj, callback)
        assert isinstance(pv.get_projection(proj), pd.DataFrame)

        #get a pv that needs to be computed
        proj = Proj(2, 2)
        assert isinstance(pv.get_projection(proj), pd.DataFrame)

    def test_is_present(self):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        assert not pv.is_present(proj)

        pv.compute(proj, callback)
        assert pv.is_present(proj)

    def test_compute(self):
        callback = DummyCallable()
        pv = ProjectedValues(self.X, self.y)
        proj = Proj(1, 2)
        pv.compute(proj, callback)
        assert isinstance(pv._projected_values[proj], pd.DataFrame)
