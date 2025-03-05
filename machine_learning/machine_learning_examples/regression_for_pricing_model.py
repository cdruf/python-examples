import numpy as np
import scipy.optimize as opt


def demand_model(p, alpha, d):
    """
    Compute the demand vector given the prices, alpha, and the total demand.

    :param p: Array of prices for different months
    :param alpha: Price sensitivity parameter
    :param d: Total demand.
    """
    exp_term = np.exp(-alpha * p)
    return d * (exp_term / np.sum(exp_term))


# Generate data: 12 months, given prices, demand according to model + noise
np.random.seed(42)
true_alpha = 0.5  # True sensitivity
d = 1000  # Total demand
p_true = np.array([10, 12, 15, 14, 11, 13, 15, 14, 9, 13, 12, 9])
d_true = demand_model(p_true, true_alpha, d) + np.random.normal(0, 5, size=len(p_true))


# Objective function
def fit_func(p, alpha):
    return demand_model(p, alpha, d)


# Use curve_fit to estimate alpha
alpha_est, _ = opt.curve_fit(fit_func, p_true, d_true, p0=[0.1])

print(f"Estimated alpha: {alpha_est[0]:.4f}")


# %%
# Model with multiple samples

def demand_model(p, alpha, d):
    """
    Computes demand matrix given prices, alpha, and the total demand.

    :param p: 2D-matrix where each row is a price vector.
    :param alpha: Price sensitivity parameter
    :param d: Total demand

    :return: demand matrix where each row .
    """
    exp_term = np.exp(-alpha * p)  # element-wise exponential
    row_sums = np.sum(exp_term, axis=1, keepdims=True)
    # Divide the elements of each row by its row sum > normalize
    return d * (exp_term / row_sums)


# Generate data: 12 months, given price vectors, demand matrix where each row is a demand vector
np.random.seed(42)
true_alpha = 0.5
d = 1000  # Total demand

p_true = np.array([
    [10, 12, 15, 14, 11, 13, 10, 12, 15, 14, 11, 13],
    [11, 13, 14, 12, 10, 15, 10, 12, 15, 14, 11, 13],
    [14, 10, 12, 15, 13, 11, 10, 12, 15, 13, 11, 13],
    [15, 14, 13, 12, 11, 10, 10, 12, 15, 12, 11, 13],
    [12, 11, 10, 13, 14, 15, 10, 12, 9, 14, 11, 13]
])  # each row is a price vector
d_true = demand_model(p_true, true_alpha, d) + np.random.normal(0, 5, p_true.shape)


# Objective function for curve fitting
def fit_func(p_flat, alpha):
    """
    """
    p = p_flat.reshape(-1, p_true.shape[1])  # Reshape back into 2D
    return demand_model(p, alpha, d).flatten()  # Flatten output to match demand vector


# Flatten inputs for curve fitting
p_flat = p_true.flatten()
d_flat = d_true.flatten()

# Estimate alpha
alpha_est, _ = opt.curve_fit(fit_func, p_flat, d_flat, p0=[0.1])

print(f"Estimated alpha: {alpha_est[0]:.4f}")
