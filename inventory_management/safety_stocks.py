import scipy

from inventory_management.loss_functions import loss_function_standard_normal

EPS = 1e-4


def get_reorder_point_and_safety_stock_level(mu, std, target_fill_rate=0.95):
    # Calculate target units short
    target_units_short = (1.0 - target_fill_rate) * mu

    # Find optimal reorder point
    def f(x):
        return abs(loss_function_standard_normal(x) * std - target_units_short)

    result = scipy.optimize.minimize(f, x0=(0.0,))

    ss = result.x.item() * std
    rop = mu + ss
    return rop, ss
