r"""
The final step is to complete the function `estimate_bispectrum_binned` which acts as the main orchestrator and uses earlier steps to generate a PNG Fourier field and estimate a binned bispectrum.

Given `N`, `box_size`, spectrum parameters `A_s`, `n_s`, `k_pivot`, RNG `seed`, PNG amplitude `f_NL`, and Schwinger controls `Nt`, `t_max`, `t_max_factor`, construct the pipeline by calling the functions defined in previous steps. To do this, you should use:
- `kgrid_3d(N, box_size)` to get `k_mag`; `primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)` to get `P_phi`;
- `generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero=True)` to get `phiG_k`; 
- `equilateral_bispectrum_factors(P_phi)` to get `b1, b2, b3`; 
- `phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)` to get `phiNG_k`,

and finally set $\phi(\mathbf{k}) = \phi_G(\mathbf{k}) + f_{\rm NL}\,\phi_{\rm NG}(\mathbf{k})$. 

The function should estimate the binned bispectrum by averaging $\phi(\mathbf{q}_1)\phi(\mathbf{q}_2)\phi(\mathbf{q}_3)$ over wrapped closed triangles whose magnitudes satisfy $||\mathbf{q}_i|-k_i|\le \Delta k/2$ with `dk` as $\Delta k$, enforcing closure by index arithmetic $\text{idx}_3 = (-\text{idx}_1-\text{idx}_2)\bmod N$.

Require `N` to be a positive integer, `box_size > 0`, `A_s >= 0` and `k_pivot > 0` (raise `ValueError` on violation, either directly or via the called functions from previous steps). At the end, the function should return `B_hat` (complex scalar) and `N_tri` (int), and if `N_tri == 0` return `(0j, 0)`.
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def estimate_bispectrum_binned(N: int, box_size: float, k1: float, k2: float, k3: float, dk: float, *, A_s: float = 1.0, n_s: float = 1.0, k_pivot: float = 1.0, seed: int = 0, f_NL: float = 0.0, Nt: int = 32, t_max: Optional[float] = None, t_max_factor: float = 8.0,) -> Tuple[complex, int]:
    """
    Orchestrate PNG field generation and estimate a binned bispectrum on an FFT grid.

    Parameters
    ----------
    N: int
        Grid size per dimension.
    box_size: float
        Side length L of the periodic box.
    k1, k2, k3: float
        Target magnitudes for the three triangle legs.
    dk: float
        Bin width (half-width is dk/2).

    Other Parameters
    ----------------
    A_s, n_s, k_pivot: float
        Power spectrum parameters forwarded to `primordial_power_spectrum`.
    seed: int
        RNG seed forwarded to `generate_gaussian_potential_fourier`.
    f_NL: float
        PNG amplitude used to form `phi_k = phiG_k + f_NL * phiNG_k`.
    Nt, t_max, t_max_factor:
        Schwinger integration parameters forwarded to `phi_ng_schwinger`.

    Returns
    -------
    B_hat: complex
        Complex average triangle product in the requested bins.
    N_tri: int
        Number of triangles contributing to the average.
    """
    return B_hat, N_tri  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

# Local copies of helper functions defined in previous steps (to keep this step self contained)

def _gold_kgrid_3d_local(N: int, box_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(N) != N or N <1:
        raise ValueError("N must be a positive integer")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")
    
    k1d = np.fft.fftfreq(N, d=box_size / N) * (2.0 * np.pi)
    kx,ky,kz= np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_mag = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, k_mag


def _gold_primordial_power_spectrum_local(k_mag: np.ndarray, A_s: float, n_s: float, k_pivot: float) -> np.ndarray:
    if k_pivot <= 0:
        raise ValueError("K_pivot must be > 0")
    if A_s < 0:
        raise ValueError("A_s must be >= 0")

    k = np.asarray(k_mag, dtype=float)
    if np.any(k<0):
        raise ValueError("k_mag must be non-negative")
    
    P_phi=np.zeros_like(k, dtype=float)
    mask = k > 0.0
    C = 18.0 * np.pi**2 / 25.0
    P_phi[mask]= C * A_s * (k[mask]**(n_s-4.0)) * (k_pivot ** (1.0 - n_s))
    return P_phi 

def _gold_generate_gaussian_potential_fourier_local(P_phi: np.ndarray, seed: int, set_k0_zero: bool) -> np.ndarray:
    P = np.asarray(P_phi, dtype=float)
    if P.ndim != 3 or P.shape[0] != P.shape[1] or P.shape[0] != P.shape[2]:
        raise ValueError("P_phi must have shape (N,N,N)")
    if np.any(P<0):
        raise ValueError("P_phi must be non-negative")
    N = P.shape[0]
    rng = np.random.default_rng(seed)
    white_x = rng.normal(size=P.shape)
    white_k = np.fft.fftn(white_x)
    phiG_k = white_k * np.sqrt(P) / np.sqrt(N**3)
    if set_k0_zero:
        phiG_k[0,0,0] = 0.0 + 0.0j
    return phiG_k

def _gold_equilateral_bispectrum_factors_local(P_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = np.asarray(P_phi, dtype=float)
    if P.ndim != 3 or P.shape[0] != P.shape[1] or P.shape[0] != P.shape[2]:
        raise ValueError("P_phi must have shape (N,N,N)")
    if np.any(P<0):
        raise ValueError("P_phi must be non-negative")
    
    # Coefficients and exponents for the 10 term decomposition of Bbar_eq = B_eq/(2 f_NL)
    coeffs = np.array([-3, -3, -3, -6, 3, 3, 3, 3, 3, 3], dtype=float)
    powers = np.array( [ [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [2.0/3.0, 2.0/3.0, 2.0/3.0], [1.0/3.0, 2.0/3.0, 1.0], [1.0/3.0, 1.0, 2.0/3.0], [2.0/3.0, 1.0/3.0, 1.0], [2.0/3.0, 1.0, 1.0/3.0], [1.0, 1.0/3.0, 2.0/3.0], [1.0, 2.0/3.0, 1.0/3.0],], dtype=float,)
    # precompute the required powers of P (including p=0 -> 1)
    unique_p= np.unique(powers)
    Pp = {}
    for p in unique_p:
        if p == 0.0:
            Pp[p] = np.ones_like(P)
        else:
            Pp[p] = P**p

    Ni = coeffs.size
    b1 = np.empty((Ni,) + P.shape, dtype=float)
    b2 = np.empty_like(b1)
    b3 = np.empty_like(b1)

    for i in range (Ni):
        c = coeffs[i]
        p1, p2, p3 = powers[i]
        b1[i] = c *Pp[p1]
        b2[i] = Pp[p2]
        b3[i] = Pp[p3]

    return b1, b2, b3

def _neg_indexer_local(N: int) -> np.ndarray:
    return (-np.arange(N)) % N

def _conv_plus_fft_local(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    idx = _neg_indexer_local(N)
    A_flip = A[np.ix_(idx, idx, idx)]
    C = (N**3) * np.fft.fftn(np.fft.ifftn(A_flip) * np.fft.ifftn(B))
    return C

def _gold_phi_ng_schwinger_local(phiG_k: np.ndarray, P_phi: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, Nt: int = 32, t_max: Optional[float] = None, t_max_factor: float = 8.0) -> np.ndarray:
    phiG_k = np.asarray(phiG_k)
    P = np.asarray(P_phi, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    b3 = np.asarray(b3, dtype=float)

    if phiG_k.ndim != 3 or phiG_k.shape[0] != phiG_k.shape[1] or phiG_k.shape[0] != phiG_k.shape[2]:
        raise ValueError("phiG_k must have shape (N,N,N)")
    if P.shape != phiG_k.shape:
        raise ValueError("P_phi must have the same shape as phiG_k)")
    if b1.ndim != 4 or b1.shape[1:] != phiG_k.shape:
        raise ValueError("b1 must have shape (Ni,N,N,N)")
    if b2.shape != b1.shape or b3.shape != b1.shape:
        raise ValueError("b2 and b3 must have the same shape as b1")

    N = phiG_k.shape[0]
    Ni = b1.shape[0]

    Ppos = P[P > 0.0]
    Pmax = float(Ppos.max()) if Ppos.size else 1.0
    if t_max is None:
        t_max = t_max_factor * Pmax

    # Trapezoidal weights on [0,t_max].
    if Nt < 2:
        t= np.array([0.0], dtype=float)
        w_t= np.array([float(t_max)], dtype=float)
    else:
        t = np.linspace(0.0, float(t_max), int(Nt), dtype=float)
        dt = float(t[1] - t[0])
        w_t = np.full(t.shape, dt, dtype=float)
        w_t[0] *= 0.5
        w_t[-1] *= 0.5

    mask = P > 0.0
    phiNG = np.zeros_like(phiG_k, dtype=complex)

    for tj, wt in zip(t, w_t):
        E = np.zeros_like(P, dtype=float)
        E[mask] = np.exp(-tj / P[mask]) / P[mask]

        for i in range(Ni):
            w1= b1[i] * E
            A= (b2[i] * E) * np.conj(phiG_k)
            B= (b3[i] * E) * phiG_k
            conv = _conv_plus_fft_local(A, B)
            phiNG += wt * w1 * conv

    phiNG[0, 0, 0] = 0.0 + 0.0j
    return phiNG

# Gold solution for step 6 - self-contained reference implementation
 
def _gold_estimate_bispectrum_binned(N: int, box_size: float, k1: float, k2: float, k3: float, dk: float, *, A_s: float = 1.0, n_s: float = 1.0, k_pivot: float = 1.0, seed: int = 0, f_NL: float = 0.0, Nt: int = 32, t_max: Optional[float] = None, t_max_factor: float = 8.0,) -> Tuple[complex, int]:
    '''Reference implementation.'''
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")

    k1d = np.fft.fftfreq(N, d=box_size / N) * (2.0 * np.pi)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_mag = np.sqrt(kx * kx + ky * ky + kz * kz)

    if k_pivot <= 0:
        raise ValueError("K_pivot must be > 0")
    if A_s < 0:
        raise ValueError("A_s must be >= 0")

    k = np.asarray(k_mag, dtype=float)
    if np.any(k < 0):
        raise ValueError("k_mag must be non-negative")

    P_phi = np.zeros_like(k, dtype=float)
    mask = k > 0.0
    C = 18.0 * np.pi**2 / 25.0
    P_phi[mask] = C * A_s * (k[mask]**(n_s - 4.0)) * (k_pivot**(1.0 - n_s))

    P = np.asarray(P_phi, dtype=float)
    if P.ndim != 3 or P.shape[0] != P.shape[1] or P.shape[0] != P.shape[2]:
        raise ValueError("P_phi must have shape (N,N,N)")
    if np.any(P < 0):
        raise ValueError("P_phi must be non-negative")

    Ngrid = P.shape[0]
    rng = np.random.default_rng(seed)
    white_x = rng.normal(size=P.shape)
    white_k = np.fft.fftn(white_x)
    phiG_k = white_k * np.sqrt(P) / np.sqrt(Ngrid**3)
    phiG_k[0, 0, 0] = 0.0 + 0.0j

    coeffs = np.array([-3, -3, -3, -6, 3, 3, 3, 3, 3, 3], dtype=float)
    powers = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0, 1.0],
            [1.0 / 3.0, 1.0, 2.0 / 3.0],
            [2.0 / 3.0, 1.0 / 3.0, 1.0],
            [2.0 / 3.0, 1.0, 1.0 / 3.0],
            [1.0, 1.0 / 3.0, 2.0 / 3.0],
            [1.0, 2.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=float,
    )

    unique_p = np.unique(powers)
    Pp = {}
    for p in unique_p:
        if p == 0.0:
            Pp[p] = np.ones_like(P)
        else:
            Pp[p] = P**p

    Ni = coeffs.size
    b1 = np.empty((Ni,) + P.shape, dtype=float)
    b2 = np.empty_like(b1)
    b3 = np.empty_like(b1)

    for i in range(Ni):
        c = coeffs[i]
        p1, p2, p3 = powers[i]
        b1[i] = c * Pp[p1]
        b2[i] = Pp[p2]
        b3[i] = Pp[p3]

    phiG_k = np.asarray(phiG_k)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    b3 = np.asarray(b3, dtype=float)

    if phiG_k.ndim != 3 or phiG_k.shape[0] != phiG_k.shape[1] or phiG_k.shape[0] != phiG_k.shape[2]:
        raise ValueError("phiG_k must have shape (N,N,N)")
    if P.shape != phiG_k.shape:
        raise ValueError("P_phi must have the same shape as phiG_k)")
    if b1.ndim != 4 or b1.shape[1:] != phiG_k.shape:
        raise ValueError("b1 must have shape (Ni,N,N,N)")
    if b2.shape != b1.shape or b3.shape != b1.shape:
        raise ValueError("b2 and b3 must have the same shape as b1")

    Ni = b1.shape[0]

    Ppos = P[P > 0.0]
    Pmax = float(Ppos.max()) if Ppos.size else 1.0
    if t_max is None:
        t_max = t_max_factor * Pmax

    if Nt < 2:
        t = np.array([0.0], dtype=float)
        w_t = np.array([float(t_max)], dtype=float)
    else:
        t = np.linspace(0.0, float(t_max), int(Nt), dtype=float)
        dt = float(t[1] - t[0])
        w_t = np.full(t.shape, dt, dtype=float)
        w_t[0] *= 0.5
        w_t[-1] *= 0.5

    mask = P > 0.0
    phiNG = np.zeros_like(phiG_k, dtype=complex)
    idx = (-np.arange(Ngrid)) % Ngrid

    for tj, wt in zip(t, w_t):
        E = np.zeros_like(P, dtype=float)
        E[mask] = np.exp(-tj / P[mask]) / P[mask]

        for i in range(Ni):
            w1 = b1[i] * E
            A = (b2[i] * E) * np.conj(phiG_k)
            B = (b3[i] * E) * phiG_k
            A_flip = A[np.ix_(idx, idx, idx)]
            conv = (Ngrid**3) * np.fft.fftn(np.fft.ifftn(A_flip) * np.fft.ifftn(B))
            phiNG += wt * w1 * conv

    phiNG[0, 0, 0] = 0.0 + 0.0j
    phi_k = phiG_k + float(f_NL) * phiNG

    half = 0.5 * float(dk)
    m1 = np.abs(k_mag - float(k1)) <= half
    m2 = np.abs(k_mag - float(k2)) <= half
    m3 = np.abs(k_mag - float(k3)) <= half

    idx1 = np.argwhere(m1)
    idx2 = np.argwhere(m2)
    if idx1.size == 0 or idx2.size == 0:
        return 0.0 + 0.0j, 0

    s = 0.0 + 0.0j
    count = 0
    for a in idx1:
        ia, ja, ka = int(a[0]), int(a[1]), int(a[2])
        for b in idx2:
            ib, jb, kb = int(b[0]), int(b[1]), int(b[2])
            ic = (-ia - ib) % Ngrid
            jc = (-ja - jb) % Ngrid
            kc = (-ka - kb) % Ngrid
            if m3[ic, jc, kc]:
                s += phi_k[ia, ja, ka] * phi_k[ib, jb, kb] * phi_k[ic, jc, kc]
                count += 1

    if count == 0:
        return 0.0 + 0.0j, 0
    return s / count, count
# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
N = 4
box_size = 2*np.pi

A_s = 1.0
n_s = 1.0
k_pivot = 1.0
seed = 0
f_NL = 2.5

Nt = 3
t_max = 1.0
t_max_factor = 4.0

k1 = 1.0
k2 = 1.0
k3 = float(np.sqrt(2.0))
dk = 0.2
""",
            "call": "estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
        {
            "setup": """import numpy as np
N = 5
box_size = 1.0

A_s = 0.7
n_s = 0.96
k_pivot = 0.5
seed = 1
f_NL = 0.0

Nt = 2
t_max = None
t_max_factor = 3.0

k1 = 2*np.pi  # roughly one fundamental mode
k2 = 2*np.pi
k3 = 2*np.pi
dk = 0.5
""",
            "call": "estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
        # --- Condition: return (0j, 0) when no triangles are found ---
        {
            "setup": """import numpy as np
N = 4
box_size = 2*np.pi

A_s = 1.0
n_s = 1.0
k_pivot = 1.0
seed = 0
f_NL = 1.0

Nt = 2
t_max = 1.0
t_max_factor = 2.0

k1 = 0.123
k2 = 0.123
k3 = 0.123
dk = 0.01
""",
            "call": "estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot, seed=seed, f_NL=f_NL, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
        # --- testing the constraints specified in the prompt (invalid inputs) ---
        {
            "setup": """import numpy as np
# N must be positive integer
N = 0
box_size = 1.0
k1 = 1.0; k2 = 1.0; k3 = 1.0; dk = 0.2

def run_model():
    try:
        estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# N must be integer (reject float)
N = 3.5
box_size = 1.0
k1 = 1.0; k2 = 1.0; k3 = 1.0; dk = 0.2

def run_model():
    try:
        estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# box_size must be > 0
N = 4
box_size = 0.0
k1 = 1.0; k2 = 1.0; k3 = 1.0; dk = 0.2

def run_model():
    try:
        estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# A_s must be >= 0 (propagated from power spectrum)
N = 4
box_size = 1.0
A_s = -1.0
n_s = 1.0
k_pivot = 1.0
k1 = 1.0; k2 = 1.0; k3 = 1.0; dk = 0.2

def run_model():
    try:
        estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# k_pivot must be > 0 (propagated from power spectrum)
N = 4
box_size = 1.0
A_s = 1.0
n_s = 1.0
k_pivot = 0.0
k1 = 1.0; k2 = 1.0; k3 = 1.0; dk = 0.2

def run_model():
    try:
        estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_bispectrum_binned(N, box_size, k1, k2, k3, dk, A_s=A_s, n_s=n_s, k_pivot=k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
