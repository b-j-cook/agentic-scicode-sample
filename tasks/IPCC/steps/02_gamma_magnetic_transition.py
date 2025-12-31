
r"""The goal is to compute the number of pairs e-/e+ produced per unit energy interval per |dcosTh| per quantum for magnetic transition of multipole order l. 
Its analytical expression is:
$$gamma_{M_l} = 2*alpha/pi * p1*p2/q*(q/k)**(2*l+1)/(k**2-q**2)**2 * (1+w1*w2-(p1*p2/(q**2))*(p1+p2*\cos(Th))*(p2+p1*\cos(Th)))$$.
with alpha the fine structure constant, $p1=\sqrt(w1**2-1)$ the momentum of the electron, $p2=\sqrt(w2**2-1)$ the momentum of the positron and $q=\sqrt(p1**2+p2**2+2*p1*p2*\cos(Th))$ the total momentum.
In the case where $k^2-q^2 = 0$, $q=0$ or $k==0$, returns 0.
Implement a function to compute this quantity $gamma_{M_l}$.
"""

import math
import numpy as np
import scipy.constants as cst


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def gamma_magnetic_transition(w2, Th, k, l):
    ''' Calculation of the number of e⁺e⁻ pairs emitted for a  magnetic multipole transition.
        Parameters
        ----------
        k : float
            nucleus energy transition (in units of e⁻ mass).
        w2 : float
            energy of the e⁺ (in units of e⁻ mass).
        Th : float
            correlation angle between e⁻ and e⁺ (in radian).
        l : int
            multipole order of the magnetic transition.
        Returns
        -------
        result : float
            number of pairs per unit energy interval per unit angle.
    '''
    return result


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_gamma_magnetic_transition(w2, Th, k, l):
    '''Reference implementation'''
    fact = 2 * cst.fine_structure / math.pi
    w1 = k - w2
    p1 = math.sqrt(w1**2 - 1)
    p2 = math.sqrt(w2**2 - 1) 
    q = math.sqrt(p1**2+p2**2+2*p1*p2*math.cos(Th))
    if k**2-q**2 == 0 or q==0 or k==0:
        return 0
    fact1 = p1*p2/q * (q/k)**(2*l+1) / (k**2-q**2)**2
    fact2 = 1 + w1*w2 - p1*p2/(q**2)*(p1+p2*math.cos(Th))*(p2+p1*math.cos(Th))
    result = fact * fact1 * fact2
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
w2 = 1/me
Th = np.pi/6
k = 18.15/me # energy transition of excited $^8Be$ nucleus
l = 1
""",
            "call": "gamma_magnetic_transition(w2, Th, k, l)",
            "gold_call": "_gold_gamma_magnetic_transition(w2, Th, k, l)",
        },
        {
            "setup": """
me = 0.511 # electron mass in MeV/c²
w2 = 10/me
Th = np.pi/6
k = 20.21/me # energy transition of excited $^4He$ nucleus
l = 1
""",
            "call": "gamma_magnetic_transition(w2, Th, k, l)",
            "gold_call": "_gold_gamma_magnetic_transition(w2, Th, k, l)",
        },
        {
            "setup": """
me = 0.511 # electron mass in MeV/c²
w2 = 1/me
Th = np.pi/2
k = 0
l = 1
""",
            "call": "gamma_magnetic_transition(w2, Th, k, l)",
            "gold_call": "_gold_gamma_magnetic_transition(w2, Th, k, l)",
        },
    ]
