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


def gradient_error(x: np.ndarray, xtrue: np.ndarray, dx: float, dy: float):
    """
    Relative L2 error of the gradient (H1-seminorm error).

    ||grad(x) - grad(xtrue)||_2 / ||grad(xtrue)||_2
    """
    # Gradients (order: dy, dx)
    gx_y, gx_x = np.gradient(x, dy, dx)
    gt_y, gt_x = np.gradient(xtrue, dy, dx)

    # Gradient difference
    diff_sq = (gx_x - gt_x)**2 + (gx_y - gt_y)**2
    true_sq = gt_x**2 + gt_y**2

    num = np.sqrt(np.sum(diff_sq))
    den = np.sqrt(np.sum(true_sq))

    return num / den


import numpy as np

def highfreq_removed_energy(x: np.ndarray,
                            y: np.ndarray,
                            dx: float,
                            dy: float,
                            frac: float = 0.6,
                            use_fftshift: bool = True):
    """
    Energy removed in the *high-frequency* band.

    Parameters
    ----------
    x : ndarray (Ny, Nx)
        Original image.
    y : ndarray (Ny, Nx)
        Processed image.
    dx, dy : float
        Grid spacing in x and y.
    frac : float
        Cutoff as a fraction of the Nyquist radius in frequency space.
        frac=0.6 means "frequencies with radius > 0.6 * Nyquist_radius".
        Must be in (0, 1).
    use_fftshift : bool
        If True, compute with centered frequency grids (easier to reason about).

    Returns
    -------
    Erem_hf : float
        Absolute removed high-frequency energy = sum_{HF} |Rhat|^2.
    Erem_hf_rel_total : float
        Removed high-frequency energy relative to total energy in x: Erem_hf / sum |Xhat|^2.
    Erem_hf_rel_hf : float
        Removed high-frequency energy relative to original high-frequency energy in x:
        Erem_hf / sum_{HF} |Xhat|^2.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0, 1).")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    r = x - y
    Ny, Nx = x.shape

    # Fourier transforms
    X = np.fft.fft2(x)
    R = np.fft.fft2(r)

    # Frequency grids in cycles per unit length
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)

    if use_fftshift:
        X = np.fft.fftshift(X)
        R = np.fft.fftshift(R)
        fx = np.fft.fftshift(fx)
        fy = np.fft.fftshift(fy)

    FX, FY = np.meshgrid(fx, fy)  # shapes (Ny, Nx)
    fr = np.sqrt(FX**2 + FY**2)

    # Nyquist radius (cycles per unit length)
    nyq_x = 0.5 / dx
    nyq_y = 0.5 / dy
    nyq_r = np.sqrt(nyq_x**2 + nyq_y**2)

    cutoff = frac * nyq_r
    mask_hf = fr >= cutoff

    # Energies (Parseval: scaling cancels out in ratios, so we keep plain sums)
    Erem_hf = np.sum(np.abs(R[mask_hf])**2)
    Ex_total = np.sum(np.abs(X)**2)
    Ex_hf = np.sum(np.abs(X[mask_hf])**2)

    Erem_hf_rel_total = Erem_hf / Ex_total if Ex_total != 0 else np.nan
    Erem_hf_rel_hf = Erem_hf / Ex_hf if Ex_hf != 0 else np.nan

    return Erem_hf, Erem_hf_rel_total, Erem_hf_rel_hf


from skimage.metrics import structural_similarity as ssim_metric

def ssim(x: np.ndarray, xtrue: np.ndarray, data_range=None):
    """
    Structural Similarity Index (SSIM).

    Parameters
    ----------
    x : ndarray
        Reconstructed image.
    xtrue : ndarray
        Ground truth image.
    data_range : float or None
        If None, inferred from xtrue.

    Returns
    -------
    float
        SSIM index in [-1, 1], typically [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    xtrue = np.asarray(xtrue, dtype=np.float64)

    if data_range is None:
        data_range = xtrue.max() - xtrue.min()

    return ssim_metric(
        xtrue, x,
        data_range=data_range,
        channel_axis=None
    )

