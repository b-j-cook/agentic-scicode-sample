# Background: Attitude Dynamics of a Dual-Spin Satellite

A dual-spin satellite consists of a main bus and an internal spinning wheel (momentum wheel) that provides gyroscopic stiffness.

## Angular Momentum

The total angular momentum $\mathbf{H}$ in the body frame is:

$$
\mathbf{H} = \mathbf{I}\boldsymbol{\omega} + \mathbf{h}_w
$$

Where:
- $\mathbf{I}$ is the satellite's inertia matrix.
- $\boldsymbol{\omega}$ is the body angular velocity.
- $\mathbf{h}_w$ is the angular momentum of the momentum wheel (typically aligned with the body z-axis).

## Euler's Equation

The rate of change of angular momentum in the body frame is equal to the external torques $\boldsymbol{\tau}$:

$$
\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega} + \mathbf{h}_w) = \boldsymbol{\tau}_{mag} + \boldsymbol{\tau}_{gg}
$$

Rearranging for $\dot{\boldsymbol{\omega}}$:

$$
\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} \left( \boldsymbol{\tau}_{mag} + \boldsymbol{\tau}_{gg} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega} + \mathbf{h}_w) \right)
$$

## External Torques

1.  **Magnetic Torque**: $\boldsymbol{\tau}_{mag} = \mathbf{m}_{dipole} \times \mathbf{B}_{body}$, where $\mathbf{m}_{dipole}$ is the control input.
2.  **Gravity Gradient Torque**: $\boldsymbol{\tau}_{gg} = \frac{3\mu}{r^5} (\mathbf{r}_{body} \times \mathbf{I}\mathbf{r}_{body})$.

