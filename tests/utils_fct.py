import pandas as pd
import numpy as np


class DummyCallable:

    def __init__(self):
        self.calls = []

    def call(self, *args):
        self.calls.append(args)


class DummyModel:

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return ((X.iloc[:, 0] > 0.5) & (X.iloc[:, 1] > 0.5)).astype(int)
        return ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(int)

    def fit(self, X, y):
        return self

    def score(self, *args):
        return 1


class DummyProgress(DummyCallable):

    def __init__(self):
        super().__init__()
        self.progress = 0
        self.calls = []

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self.progress = args[0]


def dummy_mask(data: pd.DataFrame | pd.Series, random_seed: int | None = None) -> pd.Series:
    """
    Generates a random mask from a dataframe
    :param data: data frame the mask is created from
    :param random_seed: random seed
    :return: mask Series
    """
    np.random.seed(random_seed)
    if isinstance(data,pd.Series):
        return pd.Series(np.random.randint(0, 2, data.shape[0]))
    else:
        return pd.Series(np.random.randint(0, 2, data.shape[0] * data.shape[1]))
