"""
Dimension reduction implementations for AntakIA.

Provides PCA, UMAP, GeoMap projections.
PaCMAP is optional (pip install antakia-core[pacmap]).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from ...utils.splittable_callback import ProgressCallback

logger = logging.getLogger(__name__)


# ===========================================================
#         Utility functions
# ===========================================================


def get_safe_n_neighbors(n_samples: int, requested: int = 15) -> int:
    """
    Calculate safe n_neighbors for small datasets.

    UMAP and PaCMAP can crash if n_neighbors is too large relative
    to the dataset size.

    Parameters
    ----------
    n_samples : int
        Number of samples in dataset
    requested : int
        Requested number of neighbors

    Returns
    -------
    int
        Safe number of neighbors
    """
    max_safe = max(2, (n_samples - 1) // 4)
    safe = min(requested, max_safe)

    if safe < requested:
        logger.warning(
            f"n_neighbors reduced from {requested} to {safe} "
            f"for dataset of {n_samples} points"
        )

    return safe


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    dimreduc_method: int = DimReducMethod.dimreduc_method_as_int(
        'PCA')  # type: ignore
    allowed_kwargs = [
        'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power',
        'n_oversamples', 'power_iteration_normalizer', 'random_state'
    ]

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 progress_callback: ProgressCallback | None = None):
        super().__init__(self.dimreduc_method,
                         PCA,
                         dimension,
                         X,
                         progress_callback=progress_callback,
                         default_parameters={
                             'n_components': dimension,
                         })


# class TSNEwrapper(TSNE):
#
#     def fit_transform(self, X):
#         return pd.DataFrame(self.fit(X.values), index=X.index)

# class TSNEDimReduc(DimReducMethod):
#     """
#     T-SNE computation class.
#     """
#     dimreduc_method = -1  # DimReducMethod.dimreduc_method_as_int('TSNE')
#     allowed_kwargs = [
#         'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
#         'n_iter_without_progress', 'min_grad_norm', 'metric', 'metric_params',
#         'init', 'verbose', 'random_state', 'method', 'angle', 'n_jobs'
#     ]
#
#     def __init__(self,
#                  X: pd.DataFrame,
#                  dimension: int = 2,
#                  progress_callback: ProgressCallback | None = None):
#         super().__init__(self.dimreduc_method,
#                          TSNEwrapper,
#                          dimension,
#                          X,
#                          progress_callback=progress_callback,
#                          default_parameters={
#                              'n_components': dimension,
#                              'n_jobs': -1
#                          })
#
#     @classmethod
#     def parameters(cls) -> dict:
#         return {
#             'perplexity': {
#                 'type': float,
#                 'min': 5,
#                 'max': 50,
#                 'default': 12
#             },
#             'learning_rate': {
#                 'type': [float, str],
#                 'min': 10,
#                 'max': 1000,
#                 'default': 'auto'
#             }
#         }


class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    dimreduc_method: int = DimReducMethod.dimreduc_method_as_int(
        'UMAP')  # type: ignore
    allowed_kwargs = [
        'n_neighbors',
        'metric',
        'metric_kwds',
        'output_metric',
        'output_metric_kwds',
        'n_epochs',
        'learning_rate',
        'init',
        'min_dist',
        'spread',
        'low_memory',
        'n_jobs',
        'set_op_mix_ratio',
        'local_connectivity',
        'repulsion_strength',
        'negative_sample_rate',
        'transform_queue_size',
        'a',
        'b',
        'random_state',
        'angular_rp_forest',
        'target_n_neighbors',
        'target_metric',
        'target_metric_kwds',
        'target_weight',
        'transform_seed',
        'transform_mode',
        'force_approximation_algorithm',
        'verbose',
        'tqdm_kwds',
        'unique',
        'densmap',
        'dens_lambda',
        'dens_frac',
        'dens_var_shift',
        'output_dens',
        'disconnection_distance',
        'precomputed_knn',
    ]

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 progress_callback: ProgressCallback | None = None):
        import umap
        super().__init__(self.dimreduc_method,
                         umap.UMAP,
                         dimension,
                         X,
                         progress_callback=progress_callback,
                         default_parameters={
                             'n_components': dimension,
                             'n_jobs': -1
                         })

    @classmethod
    def parameters(cls) -> dict:
        return {
            'n_neighbors': {
                'type': int,
                'min': 1,
                'max': 200,
                'default': 15
            },
            'min_dist': {
                'type': float,
                'min': 0.1,
                'max': 0.99,
                'default': 0.1
            }
        }


class PaCMAPDimReduc(DimReducMethod):
    """
    PaCMAP computation class.

    Optional: requires `pip install pacmap` or `pip install antakia-core[pacmap]`.
    """
    dimreduc_method: int = DimReducMethod.dimreduc_method_as_int(
        'PaCMAP')  # type: ignore
    allowed_kwargs = [
        'n_neighbors', 'MN_ratio', 'FP_ratio', 'pair_neighbors', 'pair_MN',
        'pair_FP', 'distance', 'lr', 'num_iters', 'apply_pca', 'intermediate',
        'intermediate_snapshots', 'random_state'
    ]
    has_progress_callback = True

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 progress_callback: ProgressCallback | None = None):
        try:
            from .pacmap_progress import PaCMAP as PaCMAPModel
        except ImportError:
            raise ImportError(
                "PaCMAP is not installed. Install with: "
                "pip install pacmap  or  pip install antakia-core[pacmap]"
            )
        super().__init__(self.dimreduc_method,
                         PaCMAPModel,
                         dimension,
                         X,
                         progress_callback=progress_callback,
                         default_parameters={
                             'n_components': dimension,
                             'progress_callback': progress_callback
                         })

    @classmethod
    def parameters(cls) -> dict:
        return {
            'n_neighbors': {
                'type': int,
                'min': 1,
                'max': 200,
                'default': 15
            },
            'MN_ratio': {
                'type': float,
                'min': 0.1,
                'max': 10,
                'default': 0.5,
                'scale': 'log'
            },
            'FP_ratio': {
                'type': float,
                'min': 0.1,
                'max': 10,
                'default': 2,
                'scale': 'log'
            }
        }


class GeoMapIdentity:
    """Identity transformer that returns lat/lon as-is for map visualization."""

    def __init__(self, lat_col: str = "lat", lon_col: str = "lon", **kwargs):
        self.lat_col = lat_col
        self.lon_col = lon_col

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        lat = X[self.lat_col].values if self.lat_col in X.columns else X.iloc[:, 0].values
        lon = X[self.lon_col].values if self.lon_col in X.columns else X.iloc[:, 1].values
        return pd.DataFrame({"lat": lat, "lon": lon}, index=X.index)


class GeoMapDimReduc(DimReducMethod):
    """
    Geographic Map projection using lat/lon coordinates directly.

    Not a dimensionality reduction method per se: it maps geographic
    coordinates for visualization on a Plotly scatter_map.
    Automatically detected when the DataFrame contains columns named
    'latitude'/'lat' and 'longitude'/'lon'/'long'.
    """

    dimreduc_method: int = DimReducMethod.dimreduc_method_as_int(
        'GeoMap')  # type: ignore
    allowed_kwargs = ["lat_col", "lon_col"]
    has_progress_callback = False

    _lat_col: Optional[str] = None
    _lon_col: Optional[str] = None

    def __init__(
        self,
        X: pd.DataFrame,
        dimension: int = 2,
        progress_callback: ProgressCallback | None = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
    ):
        if lat_col is None or lon_col is None:
            lat_col, lon_col = self._detect_lat_lon_columns(X)

        self._lat_col = lat_col
        self._lon_col = lon_col

        if lat_col is None or lon_col is None:
            raise ValueError(
                "GeoMap requires latitude and longitude columns. "
                "No columns named 'lat', 'latitude', 'lon', 'longitude' found."
            )

        super().__init__(
            self.dimreduc_method,
            GeoMapIdentity,
            2,  # Always 2D for maps
            X,
            progress_callback=progress_callback,
            default_parameters={
                "lat_col": lat_col,
                "lon_col": lon_col,
            },
        )
        logger.info(f"GeoMap: Using columns lat='{lat_col}', lon='{lon_col}'")

    @staticmethod
    def _detect_lat_lon_columns(X: pd.DataFrame) -> tuple:
        """Detect latitude and longitude columns by name."""
        lat_col = None
        lon_col = None

        for col in X.columns:
            col_lower = col.lower()
            if col_lower in ["latitude", "lat"]:
                lat_col = col
            elif col_lower in ["longitude", "lon", "long"]:
                lon_col = col

        return lat_col, lon_col

    @classmethod
    def has_geo_columns(cls, X: pd.DataFrame) -> bool:
        """Check if DataFrame has lat/lon columns."""
        lat_col, lon_col = cls._detect_lat_lon_columns(X)
        return lat_col is not None and lon_col is not None

    @classmethod
    def parameters(cls) -> dict:
        return {}

    def compute(self, **kwargs) -> pd.DataFrame:
        """Return lat/lon as projection coordinates."""
        self.publish_progress(0)

        lat = self.X[self._lat_col].values
        lon = self.X[self._lon_col].values

        result = pd.DataFrame({0: lon, 1: lat}, index=self.X.index)

        self.publish_progress(100)
        return result


def _build_dim_reduc_factory() -> dict[int, type[DimReducMethod]]:
    """Build factory with available methods. PaCMAP is optional."""
    methods = [
        PCADimReduc,
        UMAPDimReduc,
        GeoMapDimReduc,
    ]

    # PaCMAP: optional dependency
    try:
        from .pacmap_progress import PaCMAP as _  # noqa: F401
        methods.append(PaCMAPDimReduc)
        logger.debug("PaCMAP available for dimension reduction")
    except ImportError:
        logger.info(
            "PaCMAP not installed — using UMAP as default. "
            "Install with: pip install pacmap"
        )

    return {dm.dimreduc_method: dm for dm in methods}


dim_reduc_factory: dict[int, type[DimReducMethod]] = _build_dim_reduc_factory()


def compute_projection(X: pd.DataFrame,
                       y: pd.Series,
                       dimreduc_method: int,
                       dimension: int,
                       progress_callback: ProgressCallback | None = None,
                       fit_sample_num=None,
                       **kwargs) -> pd.DataFrame:
    dim_reduc = dim_reduc_factory.get(dimreduc_method)

    if dim_reduc is None or not DimReducMethod.is_valid_dim_number(dimension):
        raise ValueError("Cannot compute proj method #", dimreduc_method,
                         " in ", dimension, " dimensions")

    default_kwargs = {'random_state': 9, 'fit_sample_num': fit_sample_num}
    default_kwargs.update(kwargs)
    dim_reduc_kwargs = {
        k: v
        for k, v in default_kwargs.items()
        if k in dim_reduc.allowed_kwargs or k == 'fit_sample_num'
    }
    proj_values = pd.DataFrame(
        dim_reduc(  # type:ignore
            X,  # type:ignore
            dimension,  # type:ignore
            progress_callback).compute(  # type:ignore
                **dim_reduc_kwargs).values,  # type:ignore
        index=X.index)
    return proj_values
