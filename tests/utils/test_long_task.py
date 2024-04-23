import pytest

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod


def test_init():
    with pytest.raises(ValueError):
        DimReducMethod(1, None, 2, None)
