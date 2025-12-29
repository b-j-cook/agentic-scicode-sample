# Background: MPC Simulation Orchestration

The full closed-loop simulation integrates all physical models and control logic into a single loop.

## The Simulation Loop

At each control step $k$:
1. **State Update**: The current state $x_k$ is obtained from the simulation.
2. **Successive Linearization**: The controller generates a reference trajectory by propagating the nonlinear model forward using the previous control sequence.
3. **Jacobian Calculation**: The system is linearized and discretized at each point along the horizon.
4. **Optimization**: The Quadratic Program (QP) is solved to find the optimal control sequence for the current horizon.
5. **Actuation**: The first control input $u_k$ is applied to the nonlinear dynamics.
6. **Propagation**: The state is integrated forward to $t_{k+1}$ using high-fidelity dynamics.

## Receding Horizon Principle

Although the optimizer calculates a sequence of $N$ future control inputs, only the first one is implemented. At the next time step, the horizon "recedes" (shifts forward), and the entire optimization is repeated with the latest state feedback. This feedback loop allows the MPC to correct for linearization errors, unmodeled disturbances, and magnetic field variations.

