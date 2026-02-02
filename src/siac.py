import math
import numpy as np
from scipy.special import binom
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve


def siac_cgam(moments: int, BSorder: int):
    """
    Compute the SIAC cosine-series coefficients c_gamma by enforcing
    polynomial reproduction (moment conditions).

    moments : even integer r (number of enforced moments)
    BSorder : B-spline order n (controls smoothness / dissipation)

    Returns
    -------
    cgam : array of length RS+1 with symmetric coefficients used in the cosine sum
    """
    assert moments % 2 == 0, "moments should be even!"
    RS = int(np.ceil(moments / 2))
    numspline = moments + 1
    # Define matrix to determine kernel coefficients
    # Linear system A c = b encodes the moment conditions
    A=np.zeros((numspline, numspline), dtype=float)
    for m in np.arange(numspline):
        for gam in np.arange(numspline):
            component = 0.
            for n in np.arange(m+1):
                jsum = 0.
                jsum = sum((-1)**(j + BSorder-1) * binom(BSorder-1,j) * ((j - 0.5*(BSorder-2))**(BSorder+n) - (j - 0.5*BSorder)**(BSorder+n)) for j in np.arange(BSorder))
                
                component += binom(m,n)*(gam-RS)**(m-n) * math.factorial(n)/math.factorial(n+BSorder)*jsum

            A[m, gam] = component
                
    b = np.zeros((numspline))
    b[0] = 1    # consistency (zeroth moment): integral of kernel = 1
    
    #call the lu_factor function LU = linalg.lu_factor(A)
    Piv = scipy.linalg.lu_factor(A)
    #P, L, U = scipy.linalg.lu(A)
    #solve given LU and B
    cgamtemp = scipy.linalg.lu_solve(Piv, b)
    cgam = np.zeros((RS + 1))
    for igam in np.arange(RS+1):
        cgam[igam] = cgamtemp[RS-igam]
    
    # Sanity check: coefficients should sum to ~1 (can be outcommented)
    # sumcoeff = sum(cgamtemp[n] for n in np.arange(numspline))
    # print('Sum of coefficients',sumcoeff) 
    return cgam
    
    
def siac_hat_1d(omega: np.ndarray, cgam: np.ndarray, BSorder: int, h: float):
    """
    omega: radian frequencies (should be same shape as FFT freq grid)
    h: grid spacing in the corresponding direction (dx or dy)
    """
    RS = len(cgam) - 1
    
    # dimensionless freq variable
    w = h * omega
    #w = omega
    
    # cosine sum
    cgamterm = cgam[0] * np.ones_like(w, dtype=float)
    for igam in range(1, RS + 1):
        cgamterm +=  2.0 * (cgam[igam] * np.cos(igam * w))
    
    # numpy sinc(x) = sin(pi x)/(pi x)
    # sin(omega/2)/(omega/2) = sinc(omega / (2*pi))
    
    sinc_factor = np.sinc(w / (2.0 * np.pi)) ** BSorder
    
    return sinc_factor * cgamterm


def pad_image_for_fft(img: np.ndarray, moments: int, BSorder: int, pad_mode: str = "reflect"):
    """
    Pad an image before FFT-based filtering.
    """
    # Ny, Nx = img.shape
    # Approximate SIAC support radius in grid points
    R = int(np.ceil((moments + BSorder + 1) / 2))
    pad = R + 2     # safety margin

    img_pad = np.pad(img, ((pad, pad), (pad, pad)), mode=pad_mode)
    return img_pad, pad, pad


def apply_separable_freq_filter_2d(img_pad: np.ndarray, dx: float, dy: float,
                                  Sx: np.ndarray, Sy: np.ndarray):
    """
    Apply a separable 2D frequency-domain filter to the padded image.
    """
    # FFT of padded image
    F = np.fft.fft2(img_pad)

    # Broadcast multiply to form full 2D response Sy[:,None] * Sx[None,:]
    # Shape: (Ny2, 1) * (1, Nx2) -> (Ny2, Nx2)
    F *= (Sy[:, None] * Sx[None, :])

    # Back to spatial domain. take real part (imaginary roundoff errors)
    out_pad = np.real(np.fft.ifft2(F))
    return out_pad


def apply_siac_fft_2d(img: np.ndarray, dx: float, dy: float,
                      moments: int = 6, BSorder: int = 2,
                      pad_mode: str = "reflect"):
    """
    SIAC post-processing on a voxel image using a separable FFT filter.

    Steps:
      1) Compute SIAC coefficients c_gamma (moment-fitting).
      2) Pad image to reduce FFT wrap-around artifacts.
      3) Build radian frequency grids (2π * fftfreq) using dx, dy.
      4) Evaluate 1D SIAC transfer function in x and y.
      5) Apply separable 2D filter in Fourier domain.
      6) Crop back to original size.
    """
    # 1)
    cgam = siac_cgam(moments, BSorder)

    # 2)
    img_pad, py, px = pad_image_for_fft(img, moments=moments, BSorder=BSorder, pad_mode=pad_mode)
    Ny2, Nx2 = img_pad.shape

    # 3)
    omegax = 2.0 * np.pi * np.fft.fftfreq(Nx2, d=dx)  # shape (Nx2,)
    omegay = 2.0 * np.pi * np.fft.fftfreq(Ny2, d=dy)  # shape (Ny2,)

    # 4)
    Sx = siac_hat_1d(omegax, cgam, BSorder, h=dx)  # (Nx2,)
    Sy = siac_hat_1d(omegay, cgam, BSorder, h=dy)  # (Ny2,)
    # print("Sx(0) =", Sx[0], "Sy(0) =", Sy[0], "    (both should be ~1)")

    # 5)
    out_pad = apply_separable_freq_filter_2d(img_pad, dx, dy, Sx=Sx, Sy=Sy)

    # 6)
    Ny, Nx = img.shape
    out = out_pad[py:py + Ny, px:px + Nx]
    return out

