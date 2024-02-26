import pytest

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from tests.utils_fct import generate_df_series


def test_init():
    with pytest.raises(ValueError):
        DimReducMethod(1, None, 2, None)

