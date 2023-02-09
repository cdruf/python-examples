"""
Adapted from
https://towardsdatascience.com/a-guide-to-the-regression-of-rates-and-proportions-bcfe1c35344f
and modified.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import loggamma, expit
from sklearn.datasets import make_regression

# %%
# Generate data
n = 3

X, y, coef = make_regression(n_samples=1000,
                             n_features=n,
                             n_informative=n,
                             noise=5.0,
                             coef=True,
                             random_state=42)
print("Coefficients:")
print(coef)

# Map y to the interval (0,1) using the sigmoid function
y = expit(y / 50)

plt.scatter(X[:, 0], y)
plt.show()


# %%
# Log-likelyhood function

def log_likelyhood(params, y, X):
    b = np.array(params[0:-1])  # params of linear regression
    phi = params[-1]  # phi parameter of beta distribution
    mu = expit(np.dot(X, b))

    eps = 1e-6  # avoid zero-division
    ret = - np.sum(loggamma(phi + eps)  # the log likelihood
                   - loggamma(mu * phi + eps)
                   - loggamma((1 - mu) * phi + eps)
                   + (mu * phi - 1) * np.log(y + eps)
                   + ((1 - mu) * phi - 1) * np.log(1 - y + eps))

    return ret


# %%
# Fit

# initial parameters for optimization
phi = 1
b0 = 1
x0 = np.array(([b0] * n) + [phi])

res = minimize(log_likelyhood, x0=x0, args=(y, X),
               bounds=([(None, None)] * n + [(0, None)]))

b = np.array(res.x[0:X.shape[1]])
print(f"Regression params: {b}")

# %%
# Predict

y_pred = expit(np.dot(X, b))  # predictions
plt.plot(X[:, 0], y_pred, ".")
plt.show()

# %%
# Export
all = np.concatenate([X, y.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
df = pd.DataFrame(all, columns=['A', 'B', 'C', 'y', 'y_pred'])
df.to_csv('my_beta_regression.csv', index=False)
print(f"Regression params: {b}")
df.head()
