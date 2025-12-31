r"""
Complete the function `generate_gaussian_potential_fourier` to generate a deterministic Gaussian Fourier field `phiG_k` with target power spectrum `P_phi` on an $N^3$ FFT grid. The function must draw a real-space white noise field `white_x ~ Normal(0,1)` using `np.random.default_rng(seed)`, compute `white_k = np.fft.fftn(white_x)`, then set $\Phi_G(\mathbf{k}) = \mathrm{white\_K}(\mathbf{k}) \sqrt{P_\Phi(k)}/\sqrt{N^3}$. `P_phi` should be non-negative with shape `(N,N,N)` and the function should raise `ValueError` otherwise. Also if `set_k0_zero` is `True`, you should set `phiG_k[0,0,0]=0`. Finally, the function should return `phiG_K` as a complex array of shape `(N,N,N)`.
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_gaussian_potential_fourier(P_phi: np.ndarray, seed: int, set_k0_zero: bool = True) -> np.ndarray:
    '''
    Generate a Gaussian Fourier space primordial potential Φ_G(k).
    
    Parameters
    ----------
    P_phi: np.ndarray
        Target power spectrum values on the FFT grid, Shape (N,N,N), non-negative.
    seed: int
        RNG seed for determinism.
    set_k0_zero: bool
        If True, set the k=0 mode to exactly zero.
    
    Returns
    -------
    phiG_k: np.ndarray
        Complex Fourier coefficients, shape (N,N,N). The inverse FFT is real.
    '''
    return phiG_k


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_generate_gaussian_potential_fourier(P_phi: np.ndarray, seed: int, set_k0_zero: bool = True) -> np.ndarray:
    '''Reference implementation.'''
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
P_phi = np.ones((4,4,4), dtype=float)
seed = 0
set_k0_zero = True
""",
            "call": "generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
            "gold_call": "_gold_generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
        },
        {
            "setup": """import numpy as np
P_phi = np.zeros((3,3,3), dtype=float)
seed = 123
set_k0_zero = True
""",
            "call": "generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
            "gold_call": "_gold_generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
P_phi = rng.random((4,4,4))
seed = 7
set_k0_zero = False
""",
            "call": "generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
            "gold_call": "_gold_generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)",
        },
        # --- testing the constraints specified in the prompt (invalid inputs) ---
        {
            "setup": """import numpy as np
P_phi = np.ones((2,2,2), dtype=float)
P_phi[0,0,0] = -0.1
seed = 0
set_k0_zero = True

def run_model():
    try:
        generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)
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
P_phi = np.ones((4,4), dtype=float)
seed = 0
set_k0_zero = True

def run_model():
    try:
        generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_generate_gaussian_potential_fourier(P_phi, seed, set_k0_zero)
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


