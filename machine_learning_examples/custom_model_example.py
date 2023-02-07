import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.model_selection import train_test_split


class MyModel(LinearClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns
        -------
        self
            Fitted estimator.

        """
        pass

    def predict_proba(self, X):
        """

        Args:
            X:

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        """
        n_samples, n_classes = X.shape
        ret = np.random.random(n_samples)
        return ret

    def score(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.

        """
        return 1.0

    def predict(self, X):
        """

        Args:
            X:

        Returns
        -------
        self : object
            Fitted LogisticRegressionCV estimator.

        """
        p = self.predict_proba(X)
        return (p > 0.5).astype(int)


if __name__ == "__main__":
    # Generate some data
    m = 100
    n = 10
    X = np.random.random((m, n))
    y = np.random.random(m)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    mdl = MyModel()
    mdl.fit(X_train, y_train)
    mdl.predict(X_test)
