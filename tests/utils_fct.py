import pandas as pd


class DummyCallable:

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def generate_df_series():
    X = pd.DataFrame([[4, 7, 10], [5, 8, 11], [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    return X, y


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
