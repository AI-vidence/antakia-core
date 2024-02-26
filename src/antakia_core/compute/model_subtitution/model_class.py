import pandas as pd
from sklearn.inspection import permutation_importance


class NotFittedError(Exception):
    pass


class MLModel:
    def __init__(self, model, name, fitted=False):
        self.fitted = fitted
        self.model = model
        self.name = name
        self.feature_importances_ = None

    def fit(self, X, y):
        if not self.fitted:
            res = self.model.fit(X, y)
            self.fitted = True
            return res

    def predict(self, X, *args, **kwargs):
        if self.fitted:
            pred = self.model.predict(X, *args, **kwargs)
            if isinstance(pred, (pd.DataFrame, pd.Series)):
                return pred
            else:
                return pd.Series(pred, index=X.index)
        raise NotFittedError()

    def compute_feature_importances(self, X, y, score, score_type):
        if self.feature_importances_ is None:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = pd.Series(
                    self.model.feature_importances_,
                    index=X.columns
                )
            else:
                def scorer(model, X, y):
                    y_pred = model.predict(X)
                    if score_type == 'minimize':
                        return -score(y, y_pred)
                    else:
                        return score(y, y_pred)

                fi = permutation_importance(
                    self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring=scorer
                )
                self.feature_importances_ = pd.Series(
                    fi.importances_mean,
                    index=X.columns
                )

    def fit_and_compute_fi(self, X_train, y_train, X_test, y_test, score, score_type):
        self.fit(X_train, y_train)
        self.compute_feature_importances(X_test, y_test, score, score_type)


class AvgRegressionBaseline:
    def fit(self, X, y, *args, **kwargs):
        self.mean = y.mean()

    def predict(self, X, *args, **kwargs):
        return [self.mean] * len(X)


class AvgClassificationBaseline:
    def fit(self, X, y, *args, **kwargs):
        lst = list(y)
        self.majority_class = max(lst, key=lst.count)

    def predict(self, X, *args, **kwargs):
        return [self.majority_class] * len(X)


class LinearMLModel(MLModel):

    def fit(self, X, *args, **kwargs):
        super().fit(X, *args, **kwargs)
        self.means = X.mean()

    def global_explanation(self):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        coefs['intercept'] = self.model.intercept_
        return {
            'type': 'table',
            'value': coefs
        }

    def local_explanation(self, x):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        exp = coefs * x - coefs * self.means
        return {
            'type': 'table',
            'prior': self.predict(x),
            'value': exp
        }


class GAMMLMdel(MLModel):
    def global_explanation(self):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        coefs['intercept'] = self.model.intercept_
        return {
            'type': 'table',
            'value': coefs
        }

    def local_explanation(self, x):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        exp = coefs * x - coefs * self.means
        return {
            'type': 'table',
            'prior': self.predict(x),
            'value': exp
        }
