# Background: Environmental Models

High-fidelity attitude control requires modeling the Earth's magnetic field and disturbance torques such as the gravity gradient.

## Earth's Magnetic Field

For a Low Earth Orbit (LEO) satellite, the field is often modeled using a **tilted dipole** approximation. While research papers such as Halverson & Caverly (2025) use the higher-fidelity **World Magnetic Model (WMM)**, this implementation uses the tilted dipole model for simplicity and to eliminate dependencies on external magnetic data libraries. The dipole model captures the essential $1/r^3$ drop-off and the periodic variation due to Earth's rotation, which are the primary drivers for magnetic attitude control.

### Tilted Dipole Formula
The magnetic field $\mathbf{B}$ at a position $\mathbf{r}$ is given by:

$$
\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \frac{1}{r^3} \left[ 3(\mathbf{m} \cdot \hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m} \right]
$$

### Time-Varying Nature
The dipole vector $\mathbf{m}$ is tilted by approximately $11.5^\circ$. As the Earth rotates, the orientation of $\mathbf{m}$ in the ECI frame changes:

$$
\mathbf{m}_{ECI}(t) = M \begin{bmatrix} \sin(\alpha) \cos(\phi(t)) \\ \sin(\alpha) \sin(\phi(t)) \\ \cos(\alpha) \end{bmatrix}
$$

## Disturbance Torques

### Gravity Gradient Torque
The variation in the Earth's gravitational field across the satellite's volume produces a torque:

$$
\boldsymbol{\tau}_{gg} = \frac{3\mu}{r^5} (\mathbf{r}_{body} \times \mathbf{I}\mathbf{r}_{body})
$$

Where $\mu$ is the Earth's gravitational parameter and $\mathbf{r}_{body}$ is the satellite's position in the body frame.

### Aerodynamic Drag
At Low Earth Orbit altitudes (~500km), aerodynamic drag can produce significant torque. However, for the purposes of this benchmark, aerodynamic effects are omitted to focus on the coupling between magnetic actuation and gyroscopic stiffness.

