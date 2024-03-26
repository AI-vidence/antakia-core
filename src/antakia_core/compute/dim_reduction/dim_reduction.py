from typing import Callable

from .pacmap_progress import PaCMAP
import pandas as pd
from sklearn.decomposition import PCA
from openTSNE import TSNE

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod


# ===========================================================
#         Projections / Dim Reductions implementations
# ===========================================================


class PCADimReduc(DimReducMethod):
    """
    PCA computation class.
    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('PCA')
    allowed_kwargs = [
        'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power',
        'n_oversamples', 'power_iteration_normalizer', 'random_state'
    ]

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 callback: Callable | None = None,
                 fit_sample_num: int | None = None
                 ):
        super().__init__(self.dimreduc_method,
                         PCA,
                         dimension,
                         X,
                         progress_updated=callback,
                         default_parameters={
                             'n_components': dimension,
                         })


class TSNEwrapper(TSNE):

    def fit_transform(self, X):
        return pd.DataFrame(self.fit(X.values), index=X.index)


class TSNEDimReduc(DimReducMethod):
    """
    T-SNE computation class.
    """
    dimreduc_method = -1  # DimReducMethod.dimreduc_method_as_int('TSNE')
    allowed_kwargs = [
        'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
        'n_iter_without_progress', 'min_grad_norm', 'metric', 'metric_params',
        'init', 'verbose', 'random_state', 'method', 'angle', 'n_jobs'
    ]

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 callback: Callable | None = None,
                 fit_sample_num: int | None = None
                 ):
        super().__init__(self.dimreduc_method,
                         TSNEwrapper,
                         dimension,
                         X,
                         progress_updated=callback,
                         default_parameters={
                             'n_components': dimension,
                             'n_jobs': -1
                         })

    @classmethod
    def parameters(cls) -> dict:
        return {
            'perplexity': {
                'type': float,
                'min': 5,
                'max': 50,
                'default': 12
            },
            'learning_rate': {
                'type': [float, str],
                'min': 10,
                'max': 1000,
                'default': 'auto'
            }
        }


class UMAPDimReduc(DimReducMethod):
    """
    UMAP computation class.
    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('UMAP')
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
                 callback: Callable | None = None,
                 fit_sample_num: int | None = None
                 ):
        import umap
        super().__init__(self.dimreduc_method,
                         umap.UMAP,
                         dimension,
                         X,
                         progress_updated=callback,
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

    """
    dimreduc_method = DimReducMethod.dimreduc_method_as_int('PaCMAP')
    allowed_kwargs = [
        'n_neighbors', 'MN_ratio', 'FP_ratio', 'pair_neighbors', 'pair_MN',
        'pair_FP', 'distance', 'lr', 'num_iters', 'apply_pca', 'intermediate',
        'intermediate_snapshots', 'random_state'
    ]
    has_progress_callback = True

    def __init__(self,
                 X: pd.DataFrame,
                 dimension: int = 2,
                 callback: Callable | None = None,
                 fit_sample_num: int | None = None
                 ):
        super().__init__(self.dimreduc_method,
                         PaCMAP,
                         dimension,
                         X,
                         progress_updated=callback,
                         default_parameters={
                             'n_components': dimension,
                             'progress_callback': callback
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


dim_reduc_factory: dict[int, type[DimReducMethod]] = {
    dm.dimreduc_method: dm
    for dm in [PCADimReduc, TSNEDimReduc, UMAPDimReduc, PaCMAPDimReduc]
}


def compute_projection(X: pd.DataFrame,
                       y: pd.Series,
                       dimreduc_method: int,
                       dimension: int,
                       progress_callback: Callable | None = None,
                       **kwargs) -> pd.DataFrame:
    dim_reduc = dim_reduc_factory.get(dimreduc_method)

    if dim_reduc is None or not DimReducMethod.is_valid_dim_number(dimension):
        raise ValueError("Cannot compute proj method #", dimreduc_method,
                         " in ", dimension, " dimensions")

    X_scaled = DimReducMethod.scale_value_space(X, y)

    default_kwargs = {'random_state': 9}
    default_kwargs.update(kwargs)
    dim_reduc_kwargs = {
        k: v
        for k, v in default_kwargs.items() if k in dim_reduc.allowed_kwargs or k == 'fit_sample_num'
    }
    proj_values = pd.DataFrame(
        dim_reduc(  # type:ignore
            X_scaled,  # type:ignore
            dimension,  # type:ignore
            progress_callback).compute(  # type:ignore
            **dim_reduc_kwargs).values,  # type:ignore
        index=X.index)
    return proj_values
