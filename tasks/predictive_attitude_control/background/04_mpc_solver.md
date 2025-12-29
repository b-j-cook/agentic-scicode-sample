# Background: MPC and Quadratic Programming (QP)

Model Predictive Control (MPC) converts the attitude tracking objective and physical constraints into a constrained Quadratic Program (QP) solved at each control step.

## Cost Function

We seek to minimize a quadratic cost function over a prediction horizon $N$:

$$
J = \sum_{k=0}^{N-1} (x_k^T \mathbf{Q} x_k + u_k^T \mathbf{R} u_k) + x_N^T \mathbf{P} x_N
$$

where $\mathbf{Q}$ is the state penalty matrix (penalizing attitude/rate errors) and $\mathbf{R}$ is the control penalty matrix (penalizing power consumption).

## QP Formulation

The state sequence can be expressed as a function of the initial state $x_0$ and the control sequence $U = [u_0^T, u_1^T, \dots, u_{N-1}^T]^T$. This results in a dense QP:

$$
\min_{U} \frac{1}{2} U^T \mathbf{H} U + \mathbf{g}^T U
$$

subject to:
- **Actuator Limits**: $|m_i| \le m_{max}$ for each magnetic torque rod.
- **State Constraints**: Keeping the boresight within a specific half-cone angle $\gamma$.
- **Operational Constraints**: Maintaining a minimum spin rate $\omega_{min}$ for the attitude determination system.

### Numerical Solver
While the problem is formulated as a Quadratic Program, this implementation uses the **SLSQP (Sequential Least Squares Programming)** solver from `scipy.optimize.minimize`. SLSQP is used because it is natively available in the standard SciPy library and provides robust handling of both linear constraints and bounds without requiring external C-based solvers like OSQP or Gurobi.

### Successive Linearization (SL-MPC)
To handle the nonlinear dynamics, we apply **Successive Linearization**. In each control step, the nonlinear dynamics are propagated forward using the previous iteration's control sequence to generate a reference trajectory. The dynamics and constraints are then linearized around this trajectory. While the paper (Halverson & Caverly, 2025) discusses potentially multiple iterations of this process per time step, this benchmark implementation performs **one iteration per control step**, which is generally sufficient for closed-loop stability while significantly reducing computational overhead.

## Underactuation and Magnetic Torque

A unique challenge of magnetic actuation is that the torque $\boldsymbol{\tau}$ is always orthogonal to the local magnetic field $\mathbf{B}$:

$$
\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B} = -[\mathbf{B} \times] \mathbf{m}
$$

where $[\mathbf{B} \times]$ is the skew-symmetric matrix of $\mathbf{B}$. This means that at any instant, torque cannot be produced along the direction of $\mathbf{B}$. The system is **instantaneously underactuated**. 

### Time-Varying Controllability
While the system is rank-deficient at any single moment, it is **periodically controllable**. As the satellite orbits the Earth, the direction of the magnetic field vector $\mathbf{B}$ changes in the inertial frame. 

In Model Predictive Control, the horizon $N$ is chosen to be long enough that the variation in $\mathbf{B}$ allows the optimizer to "plan" torques at different points in the orbit. By solving for a sequence of controls over the horizon, the optimizer can achieve full 3-axis control by distributing torque requirements across parts of the orbit where the field orientation is favorable.

## Pointing Constraints (Half-Cone Angle)

The mission requires pointing a body axis $\mathbf{z}_b$ at an inertial target $\mathbf{t}_{eci}$. The error angle $\theta$ is constrained by:

$$
\cos(\theta) = \mathbf{z}_b \cdot \mathbf{t}_b \ge \cos(\gamma)
$$

In the QP, this nonlinear constraint is linearized around the reference trajectory to form linear inequalities of the form $\mathbf{A}_{cons} U \le \mathbf{b}_{cons}$.

