import numpy as np
from src.siac_modal import centered_cardinal_bspline, siac_cgam
import matplotlib.pyplot as plt

import math
import scipy.linalg
from scipy.special import binom
from scipy.integrate import quad

def check_cgam_symmetry(moments, BSorder, tol=1e-12):
    cgam = siac_cgam(moments, BSorder)
    sym_err = np.max(np.abs(cgam - cgam[::-1]))
    print(f"max symmetry error in c_gamma: {sym_err:.3e}")
    return sym_err < tol

def build_siac_kernel_1d(moments, BSorder):
    """
    Return the unscaled 1D SIAC kernel K(x) as a callable.

    K(x) = sum_{gamma=-R}^R c_gamma * psi(x - gamma),
    where psi is the centered cardinal B-spline of order BSorder.
    """
    cgam = siac_cgam(moments, BSorder)
    psi = centered_cardinal_bspline(BSorder)

    R = int(np.ceil(moments / 2))
    gammas = np.arange(-R, R + 1)

    def kernel(x):
        x = np.asarray(x, dtype=float)
        val = np.zeros_like(x, dtype=float)
        for coeff, gamma in zip(cgam, gammas):
            val += coeff * psi(x - gamma)
        return val

    return kernel, cgam, gammas

def check_bspline_integral(BSorder):
    psi = centered_cardinal_bspline(BSorder)
    a = -BSorder / 2
    b =  BSorder / 2
    val, _ = quad(lambda x: psi(x), a, b)
    print(f"Integral of centered B-spline of order {BSorder}: {val:.16e}")
    return val

def check_kernel_integral(moments, BSorder):
    from scipy.integrate import quad

    kernel, _, _ = build_siac_kernel_1d(moments, BSorder)
    R = int(np.ceil(moments / 2))
    a = -(R + BSorder / 2)
    b =  (R + BSorder / 2)

    val, _ = quad(lambda x: kernel(x), a, b)
    print(f"Integral of SIAC kernel: {val:.16e}")
    return val


def build_siac_system(moments: int, BSorder: int, build_full: bool = False):
    """
    Build SIAC coefficient system.

    If build_full=True:
        returns full system for shifts -RS,...,RS and moments 0,...,r.

    If build_full=False:
        returns reduced symmetric system using only even moments and
        unknowns [c_0,c_1,...,c_RS].
    """
    assert moments % 2 == 0, "moments should be even!"

    RS = int(np.ceil(moments / 2))
    numspline = moments + 1

    # First build full system
    A_full = np.zeros((numspline, numspline), dtype=float)

    for m in range(numspline):
        for gam in range(numspline):
            component = 0.0
            shift = gam - RS

            for n in range(m + 1):
                jsum = sum(
                    (-1)**(j + BSorder - 1)
                    * binom(BSorder - 1, j)
                    * (
                        (j - 0.5 * (BSorder - 2))**(BSorder + n)
                        - (j - 0.5 * BSorder)**(BSorder + n)
                    )
                    for j in range(BSorder)
                )

                component += (
                    binom(m, n)
                    * shift**(m - n)
                    * math.factorial(n) / math.factorial(n + BSorder)
                    * jsum
                )

            A_full[m, gam] = component

    b_full = np.zeros(numspline)
    b_full[0] = 1.0

    if build_full:
        return A_full, b_full, RS

    # Reduced symmetric system
    even_rows = np.arange(0, moments + 1, 2)
    R = RS + 1

    A_red = np.zeros((R, R), dtype=float)

    # center column, shift 0
    A_red[:, 0] = A_full[even_rows, RS]

    # paired columns, shifts +/-k
    for k in range(1, RS + 1):
        A_red[:, k] = (
            A_full[even_rows, RS - k]
            + A_full[even_rows, RS + k]
        )

    b_red = b_full[even_rows]

    return A_red, b_red, RS


def solve_siac_system(A, b):
    """
    Solve A c = b using LU factorization.
    """
    piv = scipy.linalg.lu_factor(A)
    cgam = scipy.linalg.lu_solve(piv, b)
    return cgam

def symmetry_error(cgam):
    """
    Max absolute symmetry defect:
        max |cgam[i] - cgam[-1-i]|
    """
    cgam = np.asarray(cgam, dtype=float)
    return np.max(np.abs(cgam - cgam[::-1]))

def relative_symmetry_error(cgam):
    cgam = np.asarray(cgam, dtype=float)
    denom = np.max(np.abs(cgam))
    if denom == 0:
        return 0.0
    return np.max(np.abs(cgam - cgam[::-1])) / denom

def relative_residual(A, cgam, b):
    r = A @ cgam - b
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.linalg.norm(r)
    return np.linalg.norm(r) / nb

def siac_standard_kernel_diagnostics(p_values, print_coeffs=False):
    """
    Loop over DG degrees p and inspect the standard SIAC kernel:
        moments = 2*p
        BSorder = p + 1

    Prints:
    - condition number of A
    - relative residual of the solve
    - max coefficient size

    Parameters
    ----------
    p_values : iterable of int
        DG degrees to test.
    print_coeffs : bool
        If True, also print cgam for each p.

    Returns
    -------
    results : list of dict
        Diagnostic data for each p.
    """
    results = []

    print("\n" + "=" * 120)
    print("SIAC standard-kernel diagnostics")
    print("=" * 120)
    print(
        f"{'p':>3} {'moments':>8} {'BSorder':>8} "
        f"{'cond(A_red)':>16}"
        f"{'rel residual':>16} {'max|c|':>14}"
    )
    print("-" * 120)

    for p in p_values:
        moments = 2*p
        BSorder = p+1

        A, b, RS = build_siac_system(moments, BSorder)
        cgam = solve_siac_system(A, b)

        condA = np.linalg.cond(A)
        
        rel_res = relative_residual(A, cgam, b)
        max_c = np.max(np.abs(cgam))

        results.append({
            "p": p,
            "moments": moments + 1,
            "BSorder": BSorder,
            "RS": RS,
            "condA": condA,
            "rel_residual": rel_res,
            "max_abs_cgam": max_c,
            "cgam": cgam,
        })

        print(
            f"{p:3d} {moments + 1:8d} {BSorder:8d} "
            f"{condA:16.6e} {rel_res:16.6e} {max_c:14.6e}"
        )

        if print_coeffs:
            print(f"  cgam = {cgam}")

    print("-" * 120)
    return results


def summarize_siac_diagnostics(results, cond_warn=1e12, res_warn=1e-10):
    print("\nSummary warnings:")
    any_warn = False

    for row in results:
        flags = []
        if row["condA"] > cond_warn:
            flags.append(f"cond(A)>{cond_warn:.1e}")
        if row["rel_residual"] > res_warn:
            flags.append(f"rel_res>{res_warn:.1e}")

        if flags:
            any_warn = True
            print(f"p={row['p']:2d}: " + ", ".join(flags))

    if not any_warn:
        print("No warnings triggered.")
        
results = siac_standard_kernel_diagnostics(
    p_values=range(0, 13),
    print_coeffs=False
)
summarize_siac_diagnostics(results)


def plot_siac_kernel_over_mesh(moments, BSorder, h=1.0, points=2000):
    """
    Plot the SIAC kernel over a simple uniform mesh for visual intuition.
    """
    kernel, cgam, gammas = build_siac_kernel_1d(moments, BSorder)

    half_support = int(np.ceil((moments + BSorder) / 2))

    x_min = -1.1 * half_support
    x_max =  1.1 * half_support
    x = np.linspace(x_min, x_max, points)
    y = kernel(x)

    elem_edges = h * np.arange(np.floor(x_min / h), np.ceil(x_max / h) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="SIAC kernel")

    for xe in elem_edges:
        plt.axvline(xe, linewidth=0.8, alpha=0.25)

    plt.axhline(0.0, linewidth=0.8, color='k', alpha=0.4)

    for g in gammas:
        plt.axvline(g, linestyle="--", linewidth=0.8, alpha=0.4)


    plt.title(f"1D SIAC kernel over a uniform mesh (moments={moments + 1}, BSorder={BSorder})")
    plt.xlabel("Element-coordinate x")
    plt.ylabel("Kernel value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("c_gamma coefficients:")
    for g, c in zip(gammas, cgam):
        print(f"gamma = {g:2d}, c_gamma = {c: .6f}")
        
    from fractions import Fraction

    print("c_gamma coefficients (fraction form):")
    for g, c in zip(gammas, cgam):
        frac = Fraction(c).limit_denominator(1000)  # adjust max denominator
        print(f"gamma = {g:2d}, c_gamma ≈ {frac}  (≈ {float(frac):.6f})")

    print(f"\nExpected total support: [{-half_support}, {half_support}]")
    
def plot_siac_kernel_with_components(moments, BSorder, points=2000, show_component_labels=True, save_fig=False):
    cgam = siac_cgam(moments, BSorder)
    psi = centered_cardinal_bspline(BSorder)

    R = int(np.ceil(moments / 2))
    gammas = np.arange(-R, R + 1)

    half_support = int(np.ceil((moments + BSorder) / 2))

    x = np.linspace(-1.1 * half_support, 1.1 * half_support, points)
    kernel = np.zeros_like(x)

    plt.figure(figsize=(10, 4))

    for coeff, gamma in zip(cgam, gammas):
        comp = coeff * psi(x - gamma)
        kernel += comp
        label = fr"$c_{{{gamma}}}B_{BSorder}(x-{gamma})$" if show_component_labels else None
        plt.plot(x, comp, linestyle="--", alpha=0.7, label=label)

    plt.plot(x, kernel, linewidth=2.5, label="Total SIAC kernel")
    plt.axhline(0.0, linewidth=0.8, color='k', alpha=0.4)
    if save_fig == False:
        plt.title(f"SIAC kernel and shifted B-spline components (moments={moments + 1}, BSorder={BSorder})")
    plt.xlabel("x")
    plt.ylabel("Value")
    if show_component_labels:
        plt.legend(ncol=2, fontsize=9)
    else:
        plt.legend()
    plt.tight_layout()
    if save_fig == True:
        plt.savefig("figures/SIAC_kernel_visual.pdf", bbox_inches="tight", dpi=300)
    plt.show()

