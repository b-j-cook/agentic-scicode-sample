# Background: Successive Linearization MPC (SL-MPC)

Magnetically actuated systems are inherently underactuated and nonlinear, making standard linear MPC prone to divergence. 

## Impact of Successive Linearization
Linear MPC typically linearizes dynamics around a fixed target. SL-MPC improves performance by:
1.  Linearizing around a **trajectory** rather than a single point.
2.  Iteratively refining that trajectory using the previous control solution.

## The Algorithm

In each control step:
1.  **Trajectory Propagation**: Use the control sequence $\mathbf{U}_{prev}$ from the last step to propagate the full nonlinear dynamics forward in time, starting from the current state $\mathbf{x}_k$. This yields a reference trajectory $\mathbf{X}_{ref}$ and a sequence of magnetic fields $\mathbf{B}_{ref}$.
2.  **Time-Varying Linearization**: Calculate the Jacobian matrices $\mathbf{A}(t)$ and $\mathbf{B}(t)$ at each point along the reference trajectory.
3.  **QP Formulation**: Solve a Quadratic Program to find the optimal *deviation* from the previous control sequence that minimizes state error and control effort.
4.  **Receding Horizon**: Apply only the first control input $\mathbf{u}_k$, and repeat the process at the next time step.

