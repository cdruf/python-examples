import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, loggamma
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class MyLogRegression:

    def __init__(self):
        self.params = None

    @staticmethod
    def _loss(params, y, X):
        b = np.array(params)  # params of linear regression
        mu = expit(np.dot(X, b))
        ret = -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        return ret

    def fit(self, X, y):
        assert 0 <= y.min()
        assert y.max() <= 1
        m, n = X.shape
        x0 = np.array(([1] * n))
        res = minimize(MyLogRegression._loss, x0=x0, args=(y, X))
        self.params = np.array(res.x[0:n])

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

    steps = [("mylogreg", MyLogRegression())]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    print("Score:", pipeline.score(X_test, y_test))
    y_pred = pipeline.predict(X_test)

    for i in range(10):
        print(y_pred[i], y_test[i])
