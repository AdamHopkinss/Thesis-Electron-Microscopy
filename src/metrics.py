import numpy as np


def removed_energy(x: np.ndarray, y: np.ndarray):
    """
    Energy removed by an operator S, defined as r = x - y.

    Parameters
    ----------
    x : ndarray
        Original image (e.g. BP or FBP).
    y : ndarray
        Processed image (e.g. SIAC(x)).

    Returns
    -------
    Erem : float
        Absolute removed energy ||x - y||_2^2.
    Erel : float
        Relative removed energy ||x - y||_2^2 / ||x||_2^2.
    """
    r = x - y
    Erem = np.sum(r**2)
    Erel = Erem / np.sum(x**2)
    return Erem, Erel


def rel_l2_err(x: np.ndarray, xtrue: np.ndarray):
    """
    Relative L2 error with respect to ground truth.

    ||x - xtrue||_2 / ||xtrue||_2
    """
    return np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue)


def gradient_error(x, xtrue):
    ...

def highfreq_removed_energy(x, y, frac=0.6):
    ...

def ssim(x, xtrue):
    ...
