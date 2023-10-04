from scipy.stats import norm


def loss_function_standard_normal(z: float) -> float:
    """
    :param z: Safety factor = SS / STD.
    """
    return norm.pdf(z) - z * (1.0 - norm.cdf(z))


def loss_function_normal(x: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return max(mu - x, 0.0)
    return sigma * loss_function_standard_normal((x - mu) / sigma)


def units_short(demand_sig, z):
    """

    Args:
        demand_sig: demand standard deviation.
        z: safety stock factor.

    Returns: Expected units short.

    """
    return demand_sig * loss_function_standard_normal(z)
