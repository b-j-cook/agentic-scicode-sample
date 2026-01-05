# Background: Minimal Average Stiffness Threshold

The primary objective of optimal dynamical stabilization is to identify the global minimum of the average stiffness $\langle u \rangle$ required to prevent the mass from diverging.

## Threshold Determination

The average stiffness for a bang-bang profile is given by:
$$
\langle u \rangle = \frac{2t_s u^+ + (T - 2t_s) u^-}{T}
$$
The absolute minimal threshold is located at the **stability boundary**. For a fixed $u^+$ and $u^-$, minimizing the average stiffness corresponds to finding the minimal $t_s$ that maintains stability ($|\text{Tr}(\mathbf{M})| \le 2$). The global threshold is found by performing a parameter sweep over $u^-$ to identify the critical value where the stable region in $t_s$ space shrinking to a single point.

## Verification

The theoretical threshold $\langle u \rangle_{min}$ can be verified by:
1. Comparing with the ground state energy $E_0$ from the Schrödinger analogy.
2. Checking the Floquet trace condition $|\text{Tr}(\mathbf{M})| = 2$.

