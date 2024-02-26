import pandas as pd

class DummyCallable:
    def __init__(self):
        self.calls = []

    def call(self, *args):
        self.calls.append(args)


def generate_df_series():

    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    return X, y
