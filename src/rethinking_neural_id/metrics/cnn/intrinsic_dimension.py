# The following code is adapted from
# https://github.com/ansuini/IntrinsicDimDeep/blob/master/IDNN/intrinsic_dimension.py

import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model


def estimate(values, fraction=0.9, verbose=False):
    """
    Estimates the intrinsic dimension of a system of points from the distance matrix.
    """

    sorted_values = np.sort(values, axis=1, kind="quicksort")

    nearest = sorted_values[:, 1]
    second_nearest = sorted_values[:, 2]

    zeros = np.where(nearest == 0)[0]
    if verbose:
        print(f"Found n. {zeros.shape[0]} elements for which r1 = 0")
        print(zeros)

    degeneracies = np.where(nearest == second_nearest)[0]
    if verbose:
        print(f"Found n. {degeneracies.shape[0]} elements for which r1 = r2")
        print(degeneracies)

    good = np.setdiff1d(np.arange(sorted_values.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print(f"Fraction good points: {good.shape[0] / sorted_values.shape[0]}")

    nearest = nearest[good]
    second_nearest = second_nearest[good]

    n_points = int(np.floor(good.shape[0] * fraction))
    n_total = good.shape[0]
    mu = np.sort(np.divide(second_nearest, nearest), axis=None, kind="quicksort")
    empirical = np.arange(1, n_total + 1, dtype=np.float64) / n_total

    x_values = np.log(mu[:-2])
    y_values = -np.log(1 - empirical[:-2])

    regression = linear_model.LinearRegression(fit_intercept=False)
    regression.fit(x_values[0:n_points, np.newaxis], y_values[0:n_points, np.newaxis])
    correlation, pvalue = pearsonr(x_values[0:n_points], y_values[0:n_points])
    return x_values, y_values, regression.coef_[0][0], correlation, pvalue
