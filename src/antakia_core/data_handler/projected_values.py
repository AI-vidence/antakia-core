from collections import namedtuple
import pandas as pd

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia_core.compute.dim_reduction.dim_reduction import compute_projection, dim_reduc_factory

Proj = namedtuple('Proj', ['reduction_method', 'dimension'])


class ProjectedValues:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self._projected_values = {}
        self._parameters = {}

    def set_parameters(self, projection: Proj, parameters: dict):
        """
        set new parameters for a (projection method, dimension)
        remove previously computed projected value
        Parameters
        ----------
        projection_method : projection method value
        dimension: dimension
        parameters: new parameters

        Returns
        -------

        """
        assert projection.reduction_method in DimReducMethod.dimreduc_methods_as_list()
        assert projection.dimension in [2, 3]

        if self._parameters.get(projection) is None:
            self.build_default_parameters(projection)

        self._parameters[projection]['previous'] = \
            self._parameters[projection]['current'].copy()
        self._parameters[projection]['current'].update(parameters)
        del self._projected_values[projection]

    def get_parameters(self, projection: Proj):
        """
        get the value of the parameters for a (projection method, dimension)
        build it to default if needed
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        if self._parameters.get(projection) is None:
            self.build_default_parameters(projection)
        return self._parameters.get(projection)

    def build_default_parameters(self, projection: Proj):
        """
        build default parameters from DimReductionMethod.parameters()
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        current = {}
        dim_reduc_parameters = dim_reduc_factory[projection.reduction_method].parameters()
        for param, info in dim_reduc_parameters.items():
            current[param] = info['default']
        self._parameters[projection] = {
            'current': current,
            'previous': current.copy()
        }

    def get_projection(self, projection: Proj, progress_callback: callable = None):
        """
        get a projection value
        computes it if necessary
        Parameters
        ----------
        projection_method
        dimension
        progress_callback

        Returns
        -------

        """
        if not self.is_present(projection):
            self.compute(projection, progress_callback)
        return self._projected_values[projection]

    def is_present(self, projection: Proj) -> bool:
        """
        tests if the projection is already computed
        Parameters
        ----------
        projection_method
        dimension

        Returns
        -------

        """
        return self._projected_values.get(projection) is not None

    def compute(self, projection: Proj, progress_callback: callable):
        """
        computes a projection and store it
        Parameters
        ----------
        projection_method
        dimension
        progress_callback

        Returns
        -------

        """
        projected_values = compute_projection(
            self.X,
            self.y,
            projection.reduction_method,
            projection.dimension,
            progress_callback,
            **self.get_parameters(projection)['current']
        )
        self._projected_values[projection] = projected_values
