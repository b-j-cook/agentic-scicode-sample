r"""
Complete the function `primordial_power_spectrum` which calculates the primordial potential power spectrum `P_phi` on an array `k_mag` using the analytic power law $P_\Phi(k)=\frac{18\pi^2}{25} A_s k^{-3}\left(\frac{k}{K_\mathrm{pivot}}\right)^{n_s-1}$, or equivalently $P_\Phi(k)=\frac{18\pi^2}{25} A_s k^{n_s-4} K_\mathrm{pivot}^{1-n_s}$. You should set `P_phi` to zero whenever `k_mag==0` to avoid divergence and require `k_mag>=0` elementwise, `A_s>=0`, and `k_pivot>0` (raising `ValueError` otherwise). The function should return a float array with the same shape as `k_mag`.
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def primordial_power_spectrum(k_mag: np.ndarray, A_s: float, n_s: float, k_pivot: float = 1.0) -> np.ndarray:
    '''
    Compute P_Φ(k) from a power-law primordial spectrum,
    
    Parameters
    ----------
    k_mag: np.ndarray
        Array of Fourier magnitudes |k| (non-negative).
    A_s: float
        Scalar amplitude.
    n_s: float
        Scalar spectral index.
    k_pivot: float
        Pivot scale K_pivot (must be > 0).
    
    Returns
    -------
    P_phi: np.ndarray
        Array of power spectrum values with the same shape as k_mag.
    '''
    return P_phi


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_primordial_power_spectrum(k_mag: np.ndarray, A_s: float, n_s: float, k_pivot: float = 1.0) -> np.ndarray:
    '''Reference implementation.'''
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
k_mag = np.array([0.0, 1.0, 2.0])
A_s = 2.0
n_s = 1.0
k_pivot = 1.0
""",
            "call": "primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
            "gold_call": "_gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
        },
        {
            "setup": """import numpy as np
k_mag = np.array([[0.0, 0.5], [1.0, 2.0]])
A_s = 1.5
n_s = 0.96
k_pivot = 0.7
""",
            "call": "primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
            "gold_call": "_gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
        },
        {
            "setup": """import numpy as np
k_mag = np.array([[[0.0, 1.0], [1.5, 2.0]],
                  [[2.5, 3.0], [4.0, 5.0]]], dtype=float)
A_s = 0.8
n_s = 1.2
k_pivot = 1.0
""",
            "call": "primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
            "gold_call": "_gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)",
        },
        # --- testing the constraints specified in the prompt (invalid inputs) ---
        {
            "setup": """import numpy as np
k_mag = np.array([0.0, 1.0, 2.0])
A_s = 1.0
n_s = 1.0
k_pivot = 0.0

def run_model():
    try:
        primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
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
k_mag = np.array([0.0, 1.0, 2.0])
A_s = -1.0
n_s = 1.0
k_pivot = 1.0

def run_model():
    try:
        primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
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
k_mag = np.array([-1.0, 0.0, 1.0])
A_s = 1.0
n_s = 1.0
k_pivot = 1.0

def run_model():
    try:
        primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_primordial_power_spectrum(k_mag, A_s, n_s, k_pivot)
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
