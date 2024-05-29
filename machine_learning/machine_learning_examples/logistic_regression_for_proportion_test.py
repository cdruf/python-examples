import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import loggamma, expit
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# %%
def loss(params, y, X):
    b = np.array(params)  # params of linear regression
    mu = expit(np.dot(X, b))
    ret = -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    return ret


# %%
# Fit

# initial parameters for optimization
x0 = np.array(([1] * n))
res = minimize(loss, x0=x0, args=(y_train, X_train))
b = np.array(res.x[0:n])
print(f"Regression params: {b}")

# %%
# Predict

y_train_pred = expit(np.dot(X_train, b))
y_test_pred = expit(np.dot(X_test, b))
print("Train score:", r2_score(y_true=y_train, y_pred=y_train_pred))
print("Test score:", r2_score(y_true=y_test, y_pred=y_test_pred))

plt.plot(X_test[:, 0], y_test_pred, ".")
plt.show()

# %%
# Export
all = np.concatenate([X, y.reshape(-1, 1),  expit(np.dot(X, b)).reshape(-1, 1)], axis=1)
df = pd.DataFrame(all, columns=['A', 'B', 'C', 'y', 'y_pred'])
df.to_csv('my_log_regression.csv', index=False)
print(f"Regression params: {b}")

df.head()
