import numpy as np


def linear_regression(x_vals, y_vals):
    """
    Performs linear regression with two arrays
    :param x_vals:
    :type x_vals:
    :param y_vals:
    :type y_vals:
    :return: list of regression parameters [fitted_line, slope, y_shift, pearson r[
    :rtype:
    """
    r = np.corrcoef(x_vals, y_vals)[0, 1]
    A = np.vstack([x_vals, np.ones(len(x_vals))]).T
    slope, y_shift = np.linalg.lstsq(A, y_vals, rcond=None)[0]
    print(slope, y_shift)
    fitted_line = slope * x_vals + y_shift
    return [fitted_line, slope, y_shift, r]
