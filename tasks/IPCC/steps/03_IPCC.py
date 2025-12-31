
r"""The goal is to compute the Internal Pair Creation Coefficient (number of e⁺e⁻ pairs emitted per unit angle) with a mixture of transitions M1 + 0.21 E1. 
More precisely, for a given correlation angle Th, compute the coefficient:
$$IPCC = \int_1^{k-1} \int_0^\pi gamma(w2, th, k) \sin(Th) dTh dw_2$$, with $gamma(w2, th, k) = gamma_{M_1} + 0.21 * gamma_{E_1}$. 
Compute the double integral using Riemann sum with integration steps of $\Delta Th = \pi/100$ and $\Delta w_2 = (k-2)/100$.
Implement a function to compute this quantity IPCC.
"""

import math
import numpy as np
import scipy.constants as cst


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def IPCC(k):
    ''' Calculation of the Internal Pair Creation Coefficient (IPCC)
        Parameters
        ----------
        k : float
            nucleus energy transition (in units of e⁻ mass).
        Returns
        -------
        result : float
            number of e⁺e⁻ pairs emitted per unit angle for magnetic+electric multipole transitions.
    '''     
    return result


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_IPCC(k):
    '''Reference implementation'''  
    M_w2 = np.linspace(1,k-1,101)
    d_w2 = M_w2[1] - M_w2[0]
    M_Th = np.linspace(0,math.pi,101)
    d_Th = M_Th[1] - M_Th[0]
    result = np.sum([[(0.21*gamma_electric_transition(w2,Th,k,1)+gamma_magnetic_transition(w2,Th,k,1))*np.sin(Th) for Th in M_Th] for w2 in M_w2])*d_w2*d_Th
    return result


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """
me = 0.511 # electron mass in MeV/c²
k = 18.15/me # energy transition of excited $^8Be$ nucleus
""",
            "call": "IPCC(k)",
            "gold_call": "_gold_IPCC(k)",
        },
        {
            "setup": """
me = 0.511 # electron mass in MeV/c²
k = 20.21/me # energy transition of excited $^4He$ nucleus
""",
            "call": "IPCC(k)",
            "gold_call": "_gold_IPCC(k)",
        }
    ]
