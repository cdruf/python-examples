import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, loggamma
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class MyBetaRegression:

    def __init__(self):
        self.params = None

    @staticmethod
    def _log_likelyhood(params, y, X):
        b = np.array(params[0:-1])  # params of linear regression
        phi = params[-1]  # phi parameter of beta distribution
        mu = expit(np.dot(X, b))

        eps = 1e-6  # avoid zero-division
        ret = - np.sum(loggamma(phi + eps)
                       - loggamma(mu * phi + eps)
                       - loggamma((1 - mu) * phi + eps)
                       + (mu * phi - 1) * np.log(y + eps)
                       + ((1 - mu) * phi - 1) * np.log(1 - y + eps))

        return ret

    def fit(self, X, y):
        assert 0 <= y.min()
        assert y.max() <= 1

        m, n = X.shape
        phi = 1
        x0 = np.array(([1] * n) + [phi])
        res = minimize(MyBetaRegression._log_likelyhood, x0=x0, args=(y, X), bounds=([(None, None)] * n + [(0, None)]))
        self.params = np.array(res.x[0:X.shape[1]])

    def predict(self, X):
        return expit(np.dot(X, self.params))

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y_true=y, y_pred=y_pred)


if __name__ == "__main__":
    df = pd.read_csv("../my_beta_regression.csv")
    X = df.loc[:, ['A', 'B', 'C']].to_numpy()
    y = df.loc[:, 'y'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    # steps = [("betareg", MyBetaRegression())]
    # pipeline = Pipeline(steps)

    mdl = MyBetaRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    for i in range(10):
        print(y_pred[i], y_test[i])
