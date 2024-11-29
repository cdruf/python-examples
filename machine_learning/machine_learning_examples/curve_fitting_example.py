import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def linear_fkt(x, a, b):
    """Linear model."""
    return a * x + b


def quadratic(x, a, b, c):
    """Quadratic model."""
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    """Cubic model."""
    return a * x ** 3 + b * x ** 2 + c * x + d


def poly4(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def poly5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def calculate_mae(x, y, func, params):
    fitted = func(x, *params)
    return abs(fitted - y).sum() / len(x)


def fit_and_plot(x, y, func, ax, label):
    params, _ = curve_fit(func, x, y)
    print(f"MAE ({label}) = {calculate_mae(x, y, func, params)}")
    ax.plot(x, func(x, *params), linestyle="dashed", label=label)
    return params


if __name__ == "__main__":
    df = pd.read_csv("../../data/curve_fitting_example.csv")
    df.head()
    x = df.index.to_numpy()
    y = df['values'].to_numpy()
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y)
    fit_and_plot(x, y, linear_fkt, ax, label="Linear")
    fit_and_plot(x, y, quadratic, ax, label="Quadratic")
    fit_and_plot(x, y, cubic, ax, label="Cubic")
    fit_and_plot(x, y, poly4, ax, label="Poly4")
    fit_and_plot(x, y, poly5, ax, label="Poly5")
    plt.legend(loc="best")
    plt.show()
