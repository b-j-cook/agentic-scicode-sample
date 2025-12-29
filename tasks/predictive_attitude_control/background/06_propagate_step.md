# Background: State Propagation

Numerical integration of nonlinear differential equations is required to predict the future state of the satellite and to update the simulation.

## Runge-Kutta Integration (RK45)

The satellite's attitude dynamics are governed by nonlinear Euler equations and quaternion kinematics:

$$
\dot{x} = f(t, x, u, B, r)
$$

To propagate the state $x_k$ at time $t_k$ to $x_{k+1}$ at time $t_{k+1} = t_k + \Delta t$, this implementation uses the **RK45 (Dormand-Prince)** variable-step integrator. This method provides a good balance between speed and numerical accuracy for rigid body rotation.

## Quaternion Normalization

Due to numerical integration errors, the norm of the attitude quaternion $\|q\|$ may slowly drift away from 1. Since quaternions must have unit norm to represent valid rotations, the state is normalized after each propagation step:

$$
q_{normalized} = \frac{q}{\|q\|}
$$

This ensures that the rotation matrices derived from the state remain physically valid.

