import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# %%
# Example data

df = pd.read_csv("data/curve_fitting_example.csv")
df.head()
x = df.index.to_numpy()
y = df['values'].to_numpy()


# %%
# Functions


def fit_and_plot(x, y, func):
    params, _ = curve_fit(func, x, y)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, func(x, *params), color='red')
    plt.show()
    return params


def calculate_mae(x, y, func, params):
    fitted = func(x, *params)
    return abs(fitted - y)._sum() / len(x)


# %%
# Linear model


def linear_fkt(x, a, b):
    return a * x + b


params = fit_and_plot(x, y, linear_fkt)
print(f"MAE = {calculate_mae(x, y, linear_fkt, params)}")


# %%
# Quadratic model

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


params = fit_and_plot(x, y, quadratic)
print(params)
print(f"MAE = {calculate_mae(x, y, quadratic, params)}")


# %%


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


params = fit_and_plot(x, y, cubic)
print(f"MAE = {calculate_mae(x, y, cubic, params)}")


# %%
# Polynomial model

def poly4(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


params = fit_and_plot(x, y, poly4)
print(f"MAE = {calculate_mae(x, y, poly4, params)}")


# %%

def poly5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


params = fit_and_plot(x, y, poly5)
print(f"MAE = {calculate_mae(x, y, poly5, params)}")

# %%
# With initial guess
params, _ = curve_fit(quadratic, x, y, p0=[-12, 680, 4200])
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, quadratic(x, *params), color='red')
plt.show()
