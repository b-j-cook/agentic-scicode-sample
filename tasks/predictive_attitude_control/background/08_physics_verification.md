# Background: Physics Verification and Conservation Laws

To ensure the simulation remains physically consistent, we verify fundamental principles of satellite dynamics: the conservation of angular momentum (impulse-momentum balance) and the physical limitations of magnetic actuation.

## Impulse-Momentum Balance

In the Earth-Centered Inertial (ECI) frame, the total angular momentum of the satellite system $\mathbf{H}_{ECI}$ changes only due to external torques $\boldsymbol{\tau}_{ECI}$. This relationship is governed by the impulse-momentum theorem:

$$
\mathbf{H}_{ECI}(t_{k+1}) = \mathbf{H}_{ECI}(t_k) + \int_{t_k}^{t_{k+1}} \boldsymbol{\tau}_{ECI}(t) \, dt
$$

Where the total angular momentum $\mathbf{H}_{ECI}$ is the sum of the spacecraft's body momentum and the internal wheel momentum, rotated from the body frame to the ECI frame:

$$
\mathbf{H}_{ECI} = \mathbf{C}_{B2I} (\mathbf{I}\boldsymbol{\omega} + \mathbf{h}_w)
$$

Verification involves calculating the residual error between the actual momentum at the end of a simulation step and the expected momentum based on the integral of external torques (magnetic and gravity gradient). Large errors typically indicate numerical integration instability or incorrect coordinate frame transformations.

## Magnetic Torque Orthogonality

A fundamental physical property of magnetic actuation is that the torque $\boldsymbol{\tau}_{mag}$ produced by a magnetic dipole $\mathbf{m}$ in a magnetic field $\mathbf{B}$ must be strictly orthogonal to the magnetic field vector:

$$
\boldsymbol{\tau}_{mag} = \mathbf{m} \times \mathbf{B} \implies \boldsymbol{\tau}_{mag} \cdot \mathbf{B} = 0
$$

This check ensures that the control system implementation respects the physical constraint that magnetorquers cannot generate torque parallel to the local magnetic field vector. Any non-zero projection $\boldsymbol{\tau}_{mag} \cdot \mathbf{B}$ indicates a violation of the physical actuation model.

