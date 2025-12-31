r"""
 Complete the function `phi_ng_schwinger` which calculates the non-Gaussianity correction `phiNG_k` from `phiG_k`, `P_phi`, and separable template factors `b1`, `b2`, `b3` (with shape `(Ni,N,N,N)`) that define $\bar B(K_1, k_2, k_3)=\sum_i b_{1,i}(k_1)b_{2,i}(k_2)b_{3,i}(k_3)$. Use the Schwinger form of the reduced bispectrum kernel denominator by sampling $t\in[0,t_{\max}]$ with `Nt` points and trapezoidal weights; if `t_max` is `None`, set `t_max = t_max_factor * max(P_phi[P_phi>0])`. For each sample $t_j$, you should calculate $E_j(k)=e^{-t_j/P_\Phi (K)}/P_\Phi (K)$ and set $E_j=0$ where `P_phi==0`. 
 
 For each term `i`, form `A = b2[i]*E*conj(phiG_k)` and `B = b3[i]*E*phiG_K`, and then calculate the wrapped sum $C(\mathbf{k})=\sum_{\mathbf{k}^\prime}A(\mathbf{k}^\prime)B(\mathbf{k}+\mathbf{k}^\prime)$ (indices wrapped modulo `N`) via FFT by flipping `A` to `A_flip(k)=A(-k)` using the index map `(-idx) % N` and then use the circular convolution formula `C = N**3 *fftn(ifftn(A_flip) * ifftn(B))`. 
 
 The function should accumulate `phiNG_k += w_t * (b1[i]*E)*C` over all $t_j$ and `i` and return `phiNG_k` with `phiNG_k[0,0,0] = 0`.
 """

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def phi_ng_schwinger(phiG_k: np.ndarray, P_phi: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, Nt: int = 32, t_max: Optional[float] = None, t_max_factor: float = 8.0) -> np.ndarray:
    '''
    Calculate Φ_NG(k) using the Schwinger parameter separable reduced kernel method. 
    
    Parameters
    ----------
    phiG_k: np.ndarray
        Gaussian primordial potential Φ_G(k), complex array of shape (N,N,N).
    P_phi: np.ndarray
        Power spectrum grid P_Φ(k), float array of shape (N,N,N).
    b1, b2, b3: np.ndarray
        Separable bispectrum factors defining Bbar_eq(k1,k2,k3)=∑_i b1_i(k1) b2_i(k2) b3_i(k3), each with shape (Ni,N,N,N).
    Nt: int
        Number of trapezoidal t-samples on [0, t_max] (Nt >= 2 recommended).
    t_max: float or None
        Upper limit of the t integral. if None, use t_max_factor * max(P_phi[P_phi>0]).
    t_max_factor: float
        Used only when t_max is None.
    
    Returns
    -------
    phiNG_k: np.ndarray
        Non-Gaussianity correction Φ_NG(k), complex array of shape (N,N,N)
    '''
    return phiNG_k  # ← This line becomes the "return hint" shown to the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

# Helper functions

def _neg_indexer(N: int) -> np.ndarray:
    """Returns the 1D index map implementing K -> -k on an FFT ordered axis."""
    return (-np.arange(N)) % N

def _conv_plus_fft(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the wrapped sum C[k] = Σ_{k'} A[k'] * B[k + k'] using FFTs.
    """
    N = A.shape[0]
    idx = _neg_indexer(N)
    A_flip = A[np.ix_(idx, idx, idx)]  # A(-k)
    C = (N**3) * np.fft.fftn(np.fft.ifftn(A_flip) * np.fft.ifftn(B))
    return C

# golden solution function
def _gold_phi_ng_schwinger(phiG_k: np.ndarray, P_phi: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, Nt: int = 32, t_max: Optional[float] = None, t_max_factor: float = 8.0) -> np.ndarray:
    '''Reference implementation.'''
    phiG_k = np.asarray(phiG_k)
    P = np.asarray(P_phi, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    b3 = np.asarray(b3, dtype=float)

    if phiG_k.ndim != 3 or phiG_k.shape[0] != phiG_k.shape[1] or phiG_k.shape[0] != phiG_k.shape[2]:
        raise ValueError("phiG_k must have shape (N,N,N)")
    if P.shape != phiG_k.shape:
        raise ValueError("P_phi must have the same shape as phiG_k")
    if b1.ndim != 4 or b1.shape[1:] != phiG_k.shape:
        raise ValueError("b1 must have shape (Ni,N,N,N)")
    if b2.shape != b1.shape or b3.shape != b1.shape:
        raise ValueError("b2 and b3 must have the same shape as b1")

    N = phiG_k.shape[0]
    Ni = b1.shape[0]
    idx = (-np.arange(N)) % N

    Ppos = P[P > 0.0]
    Pmax = float(Ppos.max()) if Ppos.size else 1.0
    if t_max is None:
        t_max = t_max_factor * Pmax

    # Trapezoidal weights on [0,t_max].
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

    for tj, wt in zip(t, w_t):
        E = np.zeros_like(P, dtype=float)
        E[mask] = np.exp(-tj / P[mask]) / P[mask]

        for i in range(Ni):
            w1 = b1[i] * E
            A = (b2[i] * E) * np.conj(phiG_k)
            B = (b3[i] * E) * phiG_k
            A_flip = A[np.ix_(idx, idx, idx)]
            conv = (N**3) * np.fft.fftn(np.fft.ifftn(A_flip) * np.fft.ifftn(B))
            phiNG += wt * w1 * conv

    phiNG[0, 0, 0] = 0.0 + 0.0j
    return phiNG
# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# Small FFT grid
N = 6
L = 2*np.pi
kfreq = np.fft.fftfreq(N, d=L/N) * 2*np.pi
kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

# Power spectrum (same analytic model as Step 02)
A_s = 1.0
n_s = 1.0
k_pivot = 1.0
C = 18*np.pi**2/25
P_phi = np.zeros_like(k_mag)
mask = (k_mag > 0)
P_phi[mask] = C*A_s*(k_mag[mask]**(n_s - 4.0))*(k_pivot**(1.0 - n_s))

# Gaussian field in Fourier space
seed = 0
rng = np.random.default_rng(seed)
white_x = rng.normal(size=(N, N, N))
white_k = np.fft.fftn(white_x)
phiG_k = white_k * np.sqrt(P_phi) / np.sqrt(N**3)
phiG_k[0,0,0] = 0.0 + 0.0j

# Equilateral Bbar factors (10 terms)
coeffs = np.array([-3,-3,-3,-6,3,3,3,3,3,3], dtype=float)
powers = np.array([
  [1,   1,   0],
  [1,   0,   1],
  [0,   1,   1],
  [2/3, 2/3, 2/3],
  [1/3, 2/3, 1],
  [1/3, 1,   2/3],
  [2/3, 1/3, 1],
  [2/3, 1,   1/3],
  [1,   1/3, 2/3],
  [1,   2/3, 1/3],
], dtype=float)

Pp = {}
for p in np.unique(powers):
  if p == 0:
    Pp[p] = np.ones_like(P_phi)
  else:
    Pp[p] = P_phi**p

Ni = coeffs.size
b1 = np.empty((Ni, N, N, N), dtype=float)
b2 = np.empty_like(b1)
b3 = np.empty_like(b1)
for i, (c, (p1, p2, p3)) in enumerate(zip(coeffs, powers)):
  b1[i] = c * Pp[p1]
  b2[i] = Pp[p2]
  b3[i] = Pp[p3]

Nt = 6
t_max = None
t_max_factor = 4.0
""",
            "call": "phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
        {
            "setup": """import numpy as np
N = 4
P_phi = np.ones((N, N, N), dtype=float)
phiG_k = np.ones((N, N, N), dtype=complex)
Ni = 10
b1 = np.zeros((Ni, N, N, N), dtype=float)
b2 = np.zeros((Ni, N, N, N), dtype=float)
b3 = np.zeros((Ni, N, N, N), dtype=float)
Nt = 4
t_max = 1.0
t_max_factor = 2.0
""",
            "call": "phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
        {
            "setup": """import numpy as np
N = 4
P_phi = np.ones((N, N, N), dtype=float)
P_phi[0,0,0] = 0.0
P_phi[1,0,0] = 0.0

seed = 1
rng = np.random.default_rng(seed)
white_x = rng.normal(size=(N, N, N))
phiG_k = np.fft.fftn(white_x) * np.sqrt(P_phi) / np.sqrt(N**3)
phiG_k[0,0,0] = 0.0 + 0.0j

# Use Ni=1 (single term) to verify general-Ni support
b1 = P_phi[None, ...]
b2 = P_phi[None, ...]
b3 = np.ones_like(b1)

Nt = 3
t_max = 2.0
t_max_factor = 1.0
""",
            "call": "phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
            "gold_call": "_gold_phi_ng_schwinger(phiG_k, P_phi, b1, b2, b3, Nt=Nt, t_max=t_max, t_max_factor=t_max_factor)",
        },
    ]
