from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from openTSNE import TSNE

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia_core.utils.long_task import dummy_progress
from tests.dummy_datasets import generate_corner_dataset
from tests.utils_fct import DummyCallable


class TestDimReducMethod(TestCase):
    def setUp(self):
        self.callback = DummyCallable()
        X, y = generate_corner_dataset(10)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def test_init(self):
        drm = DimReducMethod(1, PCA, 2, self.X)
        assert drm.dimreduc_method == 1
        assert len(drm.default_parameters) == 0
        assert drm.dimension == 2
        assert drm.dimreduc_model is PCA
        assert drm.X.equals(self.X)
        assert drm.progress_updated is dummy_progress

        drm1 = DimReducMethod(2, PCA, 2, self.X, progress_updated=self.callback)
        assert drm1.dimreduc_method == 2
        assert len(drm1.default_parameters) == 0
        assert drm1.dimension == 2
        assert drm1.dimreduc_model is PCA
        assert drm1.progress_updated == self.callback

        drm2 = DimReducMethod(-1, PCA, 2, self.X, progress_updated=self.callback)
        assert drm2.dimreduc_method == -1
        assert len(drm2.default_parameters) == 0
        assert drm2.dimension == 2
        assert drm2.dimreduc_model is PCA
        assert drm2.progress_updated == self.callback

        with pytest.raises(ValueError):
            DimReducMethod(6, PCA, 2, self.X, progress_updated=self.callback)

        with pytest.raises(ValueError):
            DimReducMethod(2, PCA, 4, self.X, progress_updated=self.callback)

    def test_dimreduc_method_as_str(self):
        assert DimReducMethod.dimreduc_method_as_str(None) is None
        assert DimReducMethod.dimreduc_method_as_str(1) == DimReducMethod.dim_reduc_methods[0]
        with pytest.raises(ValueError):
            DimReducMethod.dimreduc_method_as_str(0)

    def test_dimreduc_method_as_int(self):
        assert DimReducMethod.dimreduc_method_as_int(None) is None
        assert DimReducMethod.dimreduc_method_as_int(DimReducMethod.dim_reduc_methods[0]) == 1
        with pytest.raises(ValueError):
            DimReducMethod.dimreduc_method_as_int('Method')

    def test_dimreduc_methods_as_list(self):
        assert (DimReducMethod.dimreduc_methods_as_list() ==
                list(range(1,len(DimReducMethod.dim_reduc_methods) + 1)))

    def test_dimreduc_methods_as_str_list(self):
        assert DimReducMethod.dimreduc_methods_as_str_list() == DimReducMethod.dim_reduc_methods

    def test_dimension_as_str(self):
        assert DimReducMethod.dimension_as_str(2) == '2D'
        assert DimReducMethod.dimension_as_str(3) == '3D'
        with pytest.raises(ValueError):
            DimReducMethod.dimension_as_str(1)

    def test_is_valid_dimreduc_method(self):
        assert not DimReducMethod.is_valid_dimreduc_method(0)
        assert DimReducMethod.is_valid_dimreduc_method(1)
        assert DimReducMethod.is_valid_dimreduc_method(
            len(DimReducMethod.dim_reduc_methods))
        assert not DimReducMethod.is_valid_dimreduc_method(
            len(DimReducMethod.dim_reduc_methods) + 1)

    def test_is_valid_dim_number(self):
        assert not DimReducMethod.is_valid_dim_number(1)
        assert DimReducMethod.is_valid_dim_number(2)
        assert DimReducMethod.is_valid_dim_number(3)

    def test_get_dimension(self):
        drm = DimReducMethod(1, PCA, 2, self.X)
        assert drm.get_dimension() == 2
        drm = DimReducMethod(1, PCA, 3, self.X)
        assert drm.get_dimension() == 3

    def test_parameters(self):
        drm = DimReducMethod(1, PCA, 2, self.X)
        assert drm.parameters() == {}

    def test_compute(self):  # ok rajouter test sur publish_progress
        drm1 = DimReducMethod(1, PCA, 2, self.X, default_parameters={'n_components': 2})
        drm1.compute()
        assert drm1.default_parameters == {'n_components': 2}

        drm2 = DimReducMethod(1, TSNE, 2, self.X, default_parameters={'n_components': 2})
        drm2.compute()
        # assert drm.

    def test_scale_value_space(self):
        np.random.seed(10)
        X = pd.DataFrame(np.random.randint(0, 100, size=(6, 3)),
                         columns=list('ABC'))
        y = X.sum(axis=1)
        drm = DimReducMethod(1, None, 2, X)
        a = drm.scale_value_space(X, y)
        expected = pd.DataFrame(
            [[-0.048086, -0.153033, 0.032684], [-0.000829, 0.350276, 0.216138],
             [0.001658, -0.200644, 0.089618], [-0.070471, 0.017004, -0.144444],
             [-0.030676, -0.180239, -0.030576], [0.148405, 0.166636, -0.163422]],
            index=list(range(0, 6)),
            columns=list('ABC'))
        assert np.round(drm.scale_value_space(X, y)[::], 6).equals(expected)
