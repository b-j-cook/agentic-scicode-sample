# Background: Linearization and Discretization

Model Predictive Control (MPC) requires a linear, discrete-time representation of the satellite's nonlinear dynamics to solve the optimization problem efficiently.

## Linearization of Attitude Dynamics

The satellite state is defined as $x = [q_e^T, \omega_e^T]^T$, where $q_e$ is the error quaternion (relative to a target) and $\omega_e$ is the angular velocity error. Euler's equations for a dual-spin satellite are:

$$
\mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega} + \mathbf{h}_w) = \boldsymbol{\tau}_{ctrl} + \boldsymbol{\tau}_{dist}
$$

Linearizing around a nominal state (e.g., $q_e = [1, 0, 0, 0]^T$ and $\omega_e = 0$) using a Taylor expansion yields a continuous-time state-space model:

$$
\dot{x}(t) \approx \mathbf{A}(t)x(t) + \mathbf{B}(t)u(t)
$$

Where:
- $\mathbf{A}(t)$ is the state transition matrix (Jacobian of the dynamics). This implementation includes the **magnetic torque derivative** with respect to attitude, which is often neglected in simpler models but is necessary for capturing the coupling between the magnetic field and the body orientation.
- $\mathbf{B}(t)$ is the control input matrix, which depends on the local magnetic field $\mathbf{B}_{eci}(t)$.

## Control Input Matrix and Underactuation

The control input matrix $\mathbf{B}$ maps the magnetic dipole moments $\mathbf{m}$ to the rate of change of the state. Because magnetic torque is defined by the cross product $\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}_{local}$, the resulting matrix in the error state-space has the form:

$$
\mathbf{B} = \begin{bmatrix} \mathbf{0}_{3 \times 3} \\ \mathbf{I}^{-1} (-[\mathbf{B}_{local} \times]) \end{bmatrix}
$$

where $[\mathbf{B}_{local} \times]$ is the skew-symmetric matrix of the local magnetic field. The rank of this matrix is always 2 (provided $\mathbf{B}_{local} \neq 0$). This means that at any given moment, the satellite is **instantaneously underactuated** because no torque can be produced along the axis of the local magnetic field.

## Discretization

The MPC controller operates at discrete time steps $\Delta t$. The continuous-time matrices $(\mathbf{A}, \mathbf{B})$ are converted to discrete-time $(\mathbf{A}_k, \mathbf{B}_k)$ using the matrix exponential to ensure high accuracy for the Zero-Order Hold (ZOH) assumption:

$$
\begin{bmatrix} \mathbf{A}_k & \mathbf{B}_k \\ \mathbf{0} & \mathbf{I} \end{bmatrix} = \exp \left( \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{0} & \mathbf{0} \end{bmatrix} \Delta t \right)
$$

This method is more numerically stable and accurate than simple first-order Euler approximations, especially when dealing with the fast gyroscopic dynamics of a dual-spin satellite.

## Successive Linearization (SL-MPC)

Because the system is time-varying and nonlinear, standard MPC can drift. **Successive Linearization** improves performance by re-linearizing the dynamics around the *predicted* trajectory from the previous iteration rather than always linearizing around the origin.

