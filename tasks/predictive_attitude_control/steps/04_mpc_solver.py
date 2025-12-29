"""
Stage 3: Optimal Control Synthesis - Formulate the Quadratic Program (QP) and solve the MPC step.
"""

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate):
    """
    Formulate the dense Quadratic Program (QP) matrices and linear constraints.
    
    The QP objective is min 0.5 * U.T * H * U + g.T * U.
    Constraints include actuator limits (bounds) and linear inequalities A_cons * U <= b_cons.
    
    Parameters
    ----------
    Ak_list : list of np.ndarray
        List of N discrete-time state matrices.
    Bk_list : list of np.ndarray
        List of N discrete-time control matrices.
    Q : np.ndarray
        6x6 state weighting matrix.
    R : np.ndarray
        3x3 control weighting matrix.
    x0_error : np.ndarray
        Initial error state at t=0 (6 elements).
    N : int
        Prediction horizon.
    x_ref_traj : np.ndarray
        Reference state trajectory (N+1, 7).
    target_q : np.ndarray
        Target inertial quaternion (4 elements).
    max_cone_angle : float
        Maximum allowable pointing half-cone error (deg).
    min_spin_rate : float
        Minimum allowable spin rate around boresight (rad/s).
        
    Returns
    -------
    H : np.ndarray
        (3N x 3N) Dense QP Hessian matrix.
    g : np.ndarray
        (3N,) Dense QP gradient vector.
    A_cons : np.ndarray
        (2N x 3N) Constraint matrix for pointing and spin rate.
    b_cons : np.ndarray
        (2N,) Constraint vector.
    """
    return result  # <- Return hint for the model


def _solve_mpc_step(H, g, m_max, A_cons=None, b_cons=None):
    """
    Solve the formulated QP to find the optimal control sequence.
    
    Parameters
    ----------
    H : np.ndarray
        (3N x 3N) Quadratic cost matrix.
    g : np.ndarray
        (3N,) Linear cost vector.
    m_max : float
        Maximum magnetic dipole moment per axis (A·m²).
    A_cons : np.ndarray, optional
        Linear constraint matrix (m x 3N).
    b_cons : np.ndarray, optional
        Linear constraint vector (m,).
        
    Returns
    -------
    u_opt_seq : np.ndarray
        Optimal control sequence (3N,).
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate):
    nx = 6
    nu = 3
    Sx = np.zeros((nx * N, nx))
    Su = np.zeros((nx * N, nu * N))
    A_cum = np.eye(nx)
    for i in range(N):
        A_cum = Ak_list[i] @ A_cum
        Sx[i*nx : (i+1)*nx, :] = A_cum
        for j in range(i + 1):
            if i == j:
                Su[i*nx : (i+1)*nx, j*nu : (j+1)*nu] = Bk_list[j]
            else:
                Phi = np.eye(nx)
                for k in range(i, j, -1):
                    Phi = Phi @ Ak_list[k]
                Su[i*nx : (i+1)*nx, j*nu : (j+1)*nu] = Phi @ Bk_list[j]
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)
    
    t_q0, t_q1, t_q2, t_q3 = target_q
    q_target_inv = np.array([t_q0, -t_q1, -t_q2, -t_q3])
    
    x_ref_error = np.zeros(nx * N)
    for k in range(N):
        q_ref = x_ref_traj[k+1, 0:4]
        w_ref = x_ref_traj[k+1, 4:7]
        
        # dq = q_target^-1 * q_ref
        q_e = np.array([
            q_target_inv[0]*q_ref[0] - q_target_inv[1]*q_ref[1] - q_target_inv[2]*q_ref[2] - q_target_inv[3]*q_ref[3],
            q_target_inv[0]*q_ref[1] + q_target_inv[1]*q_ref[0] + q_target_inv[2]*q_ref[3] - q_target_inv[3]*q_ref[2],
            q_target_inv[0]*q_ref[2] - q_target_inv[1]*q_ref[3] + q_target_inv[2]*q_ref[0] + q_target_inv[3]*q_ref[1],
            q_target_inv[0]*q_ref[3] + q_target_inv[1]*q_ref[2] - q_target_inv[2]*q_ref[1] + q_target_inv[3]*q_ref[0]
        ])
        
        x_ref_error[k*nx : k*nx+3] = 2.0 * q_e[1:4]
        x_ref_error[k*nx+3 : k*nx+6] = w_ref 
        
    H = 2.0 * (Su.T @ Q_bar @ Su + R_bar)
    g = 2.0 * Su.T @ Q_bar @ (Sx @ x0_error + x_ref_error)
    
    # 1. Pointing Cone Constraint: grad . delta_theta <= b
    gamma = np.radians(max_cone_angle)
    cos_gamma = np.cos(gamma)
    z_b = np.array([0.0, 0.0, 1.0])
    
    C_target = np.array([
        [1 - 2*(t_q2**2 + t_q3**2), 2*(t_q1*t_q2 + t_q0*t_q3), 2*(t_q1*t_q3 - t_q0*t_q2)],
        [2*(t_q1*t_q2 - t_q0*t_q3), 1 - 2*(t_q1**2 + t_q3**2), 2*(t_q2*t_q3 + t_q0*t_q1)],
        [2*(t_q1*t_q3 + t_q0*t_q2), 2*(t_q2*t_q3 - t_q0*t_q1), 1 - 2*(t_q1**2 + t_q2**2)]
    ])
    t_eci = C_target[2, :]
    
    # 2. Minimum Spin Rate Constraint: -delta_w_z <= w_ref_z - w_min
    A_x = np.zeros((2 * N, nx * N))
    b_x = np.zeros(2 * N)
    
    for k in range(N):
        # Pointing cone part
        q_ref = x_ref_traj[k+1, 0:4]
        q0, q1, q2, q3 = q_ref
        C_ref = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        t_ref_b = C_ref @ t_eci
        grad = np.cross(t_ref_b, z_b)
        A_x[k, k*nx : k*nx+3] = grad
        b_x[k] = t_ref_b[2] - cos_gamma
        
        # Spin rate part: -delta_w_z <= w_ref_z - min_spin_rate
        w_ref_z = x_ref_traj[k+1, 6]
        A_x[N + k, k*nx + 5] = -1.0
        b_x[N + k] = w_ref_z - min_spin_rate
        
    A_cons = A_x @ Su
    b_cons = b_x - A_x @ Sx @ x0_error
    return H, g, A_cons, b_cons

def _gold_solve_mpc_step(H, g, m_max, A_cons=None, b_cons=None):
    n_vars = g.shape[0]
    def _objective(u): return 0.5 * u.T @ H @ u + g.T @ u
    def _gradient(u): return H @ u + g
    bounds = Bounds(-m_max * np.ones(n_vars), m_max * np.ones(n_vars))
    constraints = []
    if A_cons is not None and b_cons is not None:
        constraints.append(LinearConstraint(A_cons, -np.inf, b_cons))
    u0 = np.zeros(n_vars)
    res = minimize(_objective, u0, jac=_gradient, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success: return np.zeros(n_vars)
    return res.x

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
N = 5
Ak_list = [np.eye(6) + 0.01 * np.random.randn(6, 6) for _ in range(N)]
Bk_list = [np.zeros((6, 3)) for _ in range(N)]
for i in range(N): 
    Bk_list[i][3:6, :] = np.eye(3)
    Bk_list[i] += 0.01 * np.random.randn(6, 3)
Q = np.diag([100, 100, 100, 1, 1, 1])
R = np.eye(3) * 0.1
x0_error = np.array([0.01, -0.01, 0.02, 0.001, -0.001, 0.002])
x_ref_traj = np.zeros((N+1, 7))
x_ref_traj[:, 0] = 0.9914 # 15 deg slew
x_ref_traj[:, 1] = 0.1305
target_q = np.array([1.0, 0, 0, 0])
max_cone_angle = 20.0
min_spin_rate = 0.0
""",
            "call": "formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate)",
            "gold_call": "_gold_formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate)"
        },
        {
            "setup": """
N = 3
Ak_list = [np.eye(6) for _ in range(N)]
Bk_list = [np.zeros((6, 3)) for _ in range(N)]
for i in range(N): Bk_list[i][3:6, :] = np.eye(3)
Q = np.eye(6)
R = np.eye(3)
x0_error = np.array([0.01, -0.01, 0.02, 0.005, -0.005, 0.01]) # Non-zero error
x_ref_traj = np.zeros((N+1, 7))
x_ref_traj[:, 0] = 1.0
x_ref_traj[:, 6] = 0.01  # Below min
target_q = np.array([1.0, 0, 0, 0])
max_cone_angle = 10.0
min_spin_rate = 0.05
""",
            "call": "formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate)",
            "gold_call": "_gold_formulate_qp(Ak_list, Bk_list, Q, R, x0_error, N, x_ref_traj, target_q, max_cone_angle, min_spin_rate)"
        },
        {
            "setup": """
H = np.eye(6)
g = np.array([-1.0, -1.0, -1.0, 0, 0, 0])
m_max = 0.2
A_cons = np.zeros((1, 6))
A_cons[0, 0] = 1.0 # Constraint on first control input
b_cons = np.array([-0.1]) # Tight constraint
""",
            "call": "_solve_mpc_step(H, g, m_max, A_cons, b_cons)",
            "gold_call": "_gold_solve_mpc_step(H, g, m_max, A_cons, b_cons)"
        }
    ]

