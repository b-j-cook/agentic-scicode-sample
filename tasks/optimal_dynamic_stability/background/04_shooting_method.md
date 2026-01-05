# Background: Shooting Method for Periodicity

Determining the stability boundary and optimal control parameters requires finding trajectories that satisfy the boundary conditions $x(T) = x(0)$ and $\dot{x}(T) = \dot{x}(0)$.

## Exploiting Symmetry

For the symmetric bang-bang profile $u(t) = u(T-t)$, $T$-periodic solutions can be found by enforcing simpler conditions at the half-period. Starting with the initial conditions $\mathbf{z}(0) = [x_0, 0]^T$, the solution is $T$-periodic if the velocity vanishes at $T/2$:
$$
\dot{x}(T/2; t_s) = 0
$$

## Numerical Root Finding

The "shooting" method treats the switching time $t_s \in [0, T/2]$ as the unknown. We define an objective function:
$$
f(t_s) = \mathbf{e}_2^T \mathbf{\Phi}(T/2, 0; t_s) \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$
where $\mathbf{e}_2^T = [0, 1]$. We solve $f(t_s) = 0$ using standard root-finding algorithms (e.g., Brent's method). Each root $t_s^{(n)}$ corresponds to a specific stabilization mode $n$.
