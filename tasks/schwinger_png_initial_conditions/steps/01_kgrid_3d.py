r"""
Complete the function `kgrid_3d` which given `N` and `box_size` (the box side length $L$), builds FFT-ordered wavevector grids `kx`, `ky`, `kz` (in radians/length) and their magnitude `k_mag`, each with shape `(N, N, N)`. Use $K_{\mathrm{1d}} = 2\pi \mathrm{fftfreq}(N, d=L/N)$ via `np.fft.fftfreq`, then create 3D grids with `np.meshgrid(k1d, k1d, k1d, indexing="ij")`, and finally, calculate $k=\sqrt{k_x^2+ k_y^2 + k_z^2}$. You should require `N>=1` and `box_size>0` and the function should return `kx, ky, kz, k_mag`. 
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def kgrid_3d(N: int, box_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Construct FFT-ordered #D wavevector grids for a cubic periodic box. 
    
    Parameters
    ----------
    N: int
        Grid size per dimension.
    box_size: float
        Physical side length L of the periodic box. 
    
    Returns
    -------
    kx, ky, kz, k_mag: np.ndarray
        3D arrays of shape (N,N,N) with wavevector components (radians/length)
        and magnitude k = sqrt(kx^2 + ky^2 + kz^2).
    '''
    return kx, ky, kz, k_mag


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_kgrid_3d(N: int, box_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Reference implementation.'''
    if int(N) != N or N <1:
        raise ValueError("N must be a positive integer")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")
    
    k1d = np.fft.fftfreq(N, d=box_size / N) * (2.0 * np.pi)
    kx,ky,kz= np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_mag = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, k_mag


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
""",
            "call": "kgrid_3d(N, box_size)",
            "gold_call": "_gold_kgrid_3d(N, box_size)",
        },
        {
            "setup": """import numpy as np
N = 5
box_size = 1.0
""",
            "call": "kgrid_3d(N, box_size)",
            "gold_call": "_gold_kgrid_3d(N, box_size)",
        },
        {
            "setup": """import numpy as np
N = 1
box_size = 3.0
""",
            "call": "kgrid_3d(N, box_size)",
            "gold_call": "_gold_kgrid_3d(N, box_size)",
        },
        # --- testing the constraints specified in the prompt (invalid inputs) ---
        {
            "setup": """import numpy as np
N = 0
box_size = 1.0

def run_model():
    try:
        kgrid_3d(N, box_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_kgrid_3d(N, box_size)
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
N = 4
box_size = 0.0

def run_model():
    try:
        kgrid_3d(N, box_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_kgrid_3d(N, box_size)
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
N = 3.5
box_size = 1.0

def run_model():
    try:
        kgrid_3d(N, box_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_kgrid_3d(N, box_size)
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

