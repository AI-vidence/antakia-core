import typing

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

from antakia_core.utils.long_task import LongTask
from antakia_core.utils.splittable_callback import ProgressCallback
from antakia_core.utils.correlation_coef import correlation_coef, shuffle_dataframe_and_series


class DimReducMethod(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    dimreduc_method : int, can be PCA, TSNE etc.
    dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    """

    # Class attributes methods
    dim_reduc_methods = ['PCA', 'UMAP', 'PaCMAP']
    dimreduc_method = -1

    allowed_kwargs: list[str] = []
    has_progress_callback = False

    def __init__(
        self,
        dimreduc_method: int,
        dimreduc_model: type[TransformerMixin],
        dimension: int,
        X: pd.DataFrame,
        default_parameters: dict | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        dimreduc_method : int
            Dimension reduction methods among DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP
            We store it here (not in implementation class)
        dimension : int
            Target dimension. Can be DIM_TWO or DIM_THREE
            We store it here (not in implementation class)
        X : pd.DataFrame
            Stored in LongTask instance
        progress_updated : ProgressCallback
            Stored in LongTask instance
        """
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            if dimreduc_method == -1:
                print('warning - method is not yet supported')
            else:
                raise ValueError(dimreduc_method,
                                 " is a bad dimensionality reduction method")
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        self.dimreduc_method = dimreduc_method
        if default_parameters is None:
            default_parameters = {}
        self.default_parameters = default_parameters
        self.dimension = dimension
        self.dimreduc_model = dimreduc_model
        # IMPORTANT : we set the topic as for ex 'PCA/2' or 't-SNE/3' -> subscribers have to follow this scheme
        LongTask.__init__(self, X, progress_callback)

    @classmethod
    def dimreduc_method_as_str(cls, method: int | None) -> str | None:
        if method is None:
            return None
        elif 0 < method <= len(cls.dim_reduc_methods):
            return cls.dim_reduc_methods[method - 1]
        else:
            raise ValueError(
                f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_method_as_int(cls, method: str | None) -> int | None:
        if method is None:
            return None
        try:
            i = cls.dim_reduc_methods.index(method) + 1
            return i
        except ValueError:
            raise ValueError(
                f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_methods_as_list(cls) -> list[int]:
        return list(map(lambda x: x + 1, range(len(cls.dim_reduc_methods))))

    @classmethod
    def dimreduc_methods_as_str_list(cls) -> list[str]:
        return cls.dim_reduc_methods.copy()

    @staticmethod
    def dimension_as_str(dim) -> str:
        if dim == 2:
            return "2D"
        elif dim == 3:
            return "3D"
        else:
            raise ValueError(f"{dim}, is a bad dimension")

    @classmethod
    def is_valid_dimreduc_method(cls, method: int) -> bool:
        """
        Returns True if it is a valid dimensionality reduction method.
        """
        return 0 <= method - 1 < len(cls.dim_reduc_methods)

    @staticmethod
    def is_valid_dim_number(dim: int) -> bool:
        """
        Returns True if dim is a valid dimension number.
        """
        return dim in [2, 3]

    def get_dimension(self) -> int:
        return self.dimension

    @classmethod
    def parameters(cls) -> dict[str, dict[str, typing.Any]]:
        return {}

    def compute(self,
                fit_sample_num: int | None = None,
                **kwargs) -> pd.DataFrame:
        if fit_sample_num is None or fit_sample_num > self.X.shape[0]:
            fit_sample_num = self.X.shape[0]
        self.publish_progress(0)
        kwargs['n_components'] = self.get_dimension()
        param = self.default_parameters.copy()
        param.update(kwargs)
        dim_red_model = self.dimreduc_model(**param)
        if hasattr(dim_red_model, 'fit_transform'):
            X_red = dim_red_model.fit_transform(self.X)
        elif hasattr(dim_red_model, 'fit'):
            fitted_model = dim_red_model.fit(self.X)
            if hasattr(fitted_model, 'transform'):
                X_red = fitted_model.transform(self.X)
        else:
            raise AttributeError(
                "No fit method implemented for dimensionality reduction method"
            )
        X_red = pd.DataFrame(X_red)
        self.publish_progress(100)
        return X_red

    @classmethod
    def get_scale_values(cls,
                         X: pd.DataFrame,
                         y: pd.Series,
                         progress_callback: ProgressCallback | None,
                         method='MIS'):
        mutual_info_scores = []
        if method == 'MIS':
            from sklearn.feature_selection import mutual_info_regression
            chunck_size = 20
            for i in range(0, len(X.T), chunck_size):
                chunck_mi = mutual_info_regression(
                    X.iloc[:, i:i + chunck_size], y.iloc[:])
                mutual_info_scores.append(
                    pd.Series(chunck_mi, index=X.columns[i:i + chunck_size]))
                if progress_callback is not None:
                    progress_callback(i / len(X.T) * 100)
            return pd.concat(mutual_info_scores)
        elif method == 'NCC':
            X, y = shuffle_dataframe_and_series(X, y)
            y = y + np.random.random(size=(len(X))) * 0.1 * y.std()

            for i, col in enumerate(X.columns):
                corr = correlation_coef(X[col].values, y.values)  #type: ignore
                mutual_info_scores.append(corr)
                if progress_callback is not None:
                    progress_callback(i / len(X.T) * 100)
            return pd.Series(mutual_info_scores, index=X.columns)

    @classmethod
    def scale_value_space(cls,
                          X: pd.DataFrame,
                          y: pd.Series,
                          progress_callback: ProgressCallback | None,
                          method='NCC') -> pd.DataFrame:
        """
        Scale the values in X so that it's reduced and centered and weighted with mi
        """
        std = X.std()
        std[std == 0] = 1
        mi = cls.get_scale_values(X, y, progress_callback, method)
        return (X - X.mean()) / std * mi
