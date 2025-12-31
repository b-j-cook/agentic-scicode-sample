r"""
Complete the function `equilateral_bispectrum_factors` which uses the power spectrum grid `P_phi` (with shape `(N,N,N)`), to build separable factors `b1`, `b2`, `b3` (shape `(10, N, N, N)`) so the normalised equilateral template $B_{\mathrm{eq}}(k_1,k_2,k_3)=B_{\mathrm{eq}}/(2f_{\mathrm{NL}})$ can be evaluated as $\bar B_{\mathrm{eq}}=\sum_{i=0}^9 b1[i](k_1)b2[i](k_2)b3[i](k_3)$. You should implement the fixed 10 term decomposition used in tests with `coeffs = [-3,-3,-3,-6,3,3,3,3,3,3]` and `powers = [[1,1,0],[1,0,1],[0,1,1], [2/3,2/3,2/3],[1/3,2/3,1],[1/3,1,2/3][2/3,1/3,1],[2/3,1,1/3],[1,1/3,2/3],[1,2/3,1/3]]`, storing the coefficient in `b1`. `P_phi` must be non-negative with cubic shape `(N,N,N)` and the function should raise `ValueError` if not; zeros are allowed in `P_phi` and fractional powers of 0 should evaluate to 0.
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def equilateral_bispectrum_factors(P_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Build separable factors (b1,b2,b3) for the equilateral bispectrum Bbar_eq = B_eq/(2 f_NL).
    
    Parameters 
    ----------
    P_phi: np.ndarray
        Primordial potential power spectrum grid, shape (N,N,N), non-negative. 
    
    Returns
    -------
    b1, b2, b3: np.ndarray
        Arrays of shape (10,N,N,N) such that Bbar_eq(k1,k2,k3)=∑_i b1[i](k1) b2[i](k2) b3[i](k3).
    '''
    return b1, b2, b3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_equilateral_bispectrum_factors(P_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Reference implementation.'''
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
P_phi = np.ones((2,2,2), dtype=float)
""",
            "call": "equilateral_bispectrum_factors(P_phi)",
            "gold_call": "_gold_equilateral_bispectrum_factors(P_phi)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(0)
P_phi = rng.random((3,3,3)) + 0.1
""",
            "call": "equilateral_bispectrum_factors(P_phi)",
            "gold_call": "_gold_equilateral_bispectrum_factors(P_phi)",
        },
        {
            "setup": """import numpy as np
P_phi = np.zeros((2,2,2), dtype=float)
P_phi[0,0,1] = 1.0
P_phi[0,1,0] = 2.0
""",
            "call": "equilateral_bispectrum_factors(P_phi)",
            "gold_call": "_gold_equilateral_bispectrum_factors(P_phi)",
        },
        # --- testing the constraints specified in the prompt (invalid inputs) ---
        {
            "setup": """import numpy as np
P_phi = np.ones((2,2,2), dtype=float)
P_phi[0,0,0] = -0.5

def run_model():
    try:
        equilateral_bispectrum_factors(P_phi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_equilateral_bispectrum_factors(P_phi)
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
P_phi = np.ones((2,3,2), dtype=float)

def run_model():
    try:
        equilateral_bispectrum_factors(P_phi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_equilateral_bispectrum_factors(P_phi)
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
