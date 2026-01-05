# Background: Analytic State Propagation

The evolution of the state vector $\mathbf{z}(t) = [x(t), \dot{x}(t)]^T$ in the piecewise-constant system $\ddot{x} + u(t)x = 0$ is governed by the state transition matrix $\mathbf{\Phi}(t, t_0)$.

## Transition Matrices

For an interval $dt$ with constant stiffness $u$, the mapping $\mathbf{z}(t+dt) = \mathbf{\Phi}(dt) \mathbf{z}(t)$ is determined by the sign of $u$:

### 1. Harmonic Regime ($u > 0$)
The solution is oscillatory with frequency $\omega = \sqrt{u}$:
$$
\mathbf{\Phi}(dt) = \begin{bmatrix} \cos(\omega dt) & \frac{1}{\omega}\sin(\omega dt) \\ -\omega\sin(\omega dt) & \cos(\omega dt) \end{bmatrix}
$$

### 2. Hyperbolic Regime ($u < 0$)
The solution grows exponentially with rate $\gamma = \sqrt{-u}$:
$$
\mathbf{\Phi}(dt) = \begin{bmatrix} \cosh(\gamma dt) & \frac{1}{\gamma}\sinh(\gamma dt) \\ \gamma\sinh(\gamma dt) & \cosh(\gamma dt) \end{bmatrix}
$$

### 3. Drift Regime ($u = 0$)
$$
\mathbf{\Phi}(dt) = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix}
$$

## Numerical Robustness (Large $T$)

In the large $T$ limit (or large $dt$), the hyperbolic functions $\cosh(\gamma dt)$ and $\sinh(\gamma dt)$ approach $\frac{1}{2} e^{\gamma dt}$. To prevent numerical overflow and maintain precision in the stability analysis, the propagation is reformulated using exponential sums:
$$
x(t+dt) = \frac{1}{2}\left(x_t + \frac{\dot{x}_t}{\gamma}\right)e^{\gamma dt} + \frac{1}{2}\left(x_t - \frac{\dot{x}_t}{\gamma}\right)e^{-\gamma dt}
$$
$$
\dot{x}(t+dt) = \frac{\gamma}{2}\left(x_t + \frac{\dot{x}_t}{\gamma}\right)e^{\gamma dt} - \frac{\gamma}{2}\left(x_t - \frac{\dot{x}_t}{\gamma}\right)e^{-\gamma dt}
$$
This form explicitly separates the dominant growing mode.
