# Background: Bang-Bang Stiffness Profile

Optimal dynamical stabilization of the linear system $\ddot{x} + u(t)x = 0$ is achieved when the $T$-periodic stiffness modulation $u(t)$ follows a **bang-bang** control law. Pontryagin’s Maximum Principle (PMP) dictates that for minimal average stiffness $\langle u \rangle = \frac{1}{T} \int_0^T u(t) dt$ ensuring stability, the control $u(t)$ must switch between its extreme admissible values $u^+$ and $u^-$.

## Profile Structure

For symmetric stabilization modes, the optimal control $u(t)$ within one period $[0, T]$ is characterized by a single switching time $t_s \in [0, T/2]$:

$$
u(t) = \begin{cases} 
u^+ & t \in [0, t_s] \\
u^- & t \in [t_s, T - t_s] \\
u^+ & t \in [T - t_s, T] 
\end{cases}
$$

- **$u^+ > 0$**: Positive stiffness providing local restoration.
- **$u^- < 0$**: Negative stiffness where the potential curvature is locally unstable.
- **$t_s$**: The switching parameter that optimizes the balance between stabilizing and destabilizing phases.
