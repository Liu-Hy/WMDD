import ot
import numpy as np
import torch

def compute_wasserstein_barycenter(points, k, weighted, max_iter=10, theta=0.2, tol=1e-6):
    num_real = points.shape[0]
    dimension = points.shape[1]
    wass_bary = np.random.randn(k, dimension)
    weight = np.ones(k) / k
    weight_real = np.ones(num_real) / num_real
    old_weight = None

    if weighted:
        for i in range(max_iter):
            cost_matrix = np.sum((wass_bary[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
            weight = fixed_support_barycenter(np.expand_dims(weight_real, axis=0), cost_matrix)
            if old_weight is not None and np.linalg.norm(weight - old_weight) < tol:
                weight = old_weight
                break
            old_weight = weight
            new_wass_bary = ot.lp.free_support_barycenter([points], [weight_real], wass_bary, b=weight)
            wass_bary = (1 - theta) * wass_bary + theta * new_wass_bary
    else:
        wass_bary = ot.lp.free_support_barycenter([points], [weight_real], wass_bary, b=weight)

    return wass_bary, weight


def project_simplex(x):
    """Code adopted from https://github.com/eddardd/wasserstein-barycenters/blob/main/barycenters/wasserstein.py
    Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x[x < 0] = 0
    if np.isclose(sum(x), 0):
        y = np.zeros_like(x)
    else:
        y = x.copy() / sum(x)
    return y


def fixed_support_barycenter(B, M, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False, norm='max'):
    """Fixed Support Wasserstein Barycenter

    We follow the Algorithm 1. of [1], into calculating the Wasserstein barycenter of N measures over a pre-defined
    grid :math:`\mathbf{X}`. These measures, of course, have variable sample weights :math:`\mathbf{b}_{i}`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    M : :class:`numpy.ndarray`
        Numpy array of shape (d, d), containing the pairwise distances for the support of B
    weights : :class:`numpy.ndarray`
        Numpy array or None. If None, weights are uniform. Otherwise, weights each measure in the barycenter
    eta : float
        Mirror descent step size
    numItermax : integer
        Maximum number of descent steps
    stopThr : float
        Threshold for stopping mirror descent iterations
    verbose : bool
        If true, display information about each descent step
    norm : str
        Either 'max', 'median' or 'none'. If not 'none', normalizes pairwise distances.

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """
    a = ot.unif(M.shape[0])
    a_prev = a.copy()
    weights = ot.unif(B.shape[0]) if weights is None else weights
    if norm == "max":
        _M = M / np.max(M)
    elif norm == "median":
        _M = M / np.median(M)
    else:
        _M = M

    for k in range(numItermax):
        potentials = []
        for i in range(B.shape[0]):
            _, ret = ot.emd(a, B[i], _M, log=True)
            potentials.append(ret['u'])

        # Calculates the gradient
        f_star = sum(potentials) / len(potentials)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
    return a
