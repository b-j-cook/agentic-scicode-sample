"""
Stage 2: Linearization & Discretization - Implement the Successive Linearization (SL) update logic.
This function refines the prediction by propagating the nonlinear dynamics 
using the previous iteration's control sequence to generate a new linearization 
trajectory.
"""

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params):
    """
    Perform one iteration of successive linearization by propagating nonlinear dynamics.
    
    Parameters
    ----------
    x0 : np.ndarray
        Current true state of the satellite [q, w] (7 elements).
    u_prev_seq : np.ndarray
        Control sequence from the previous MPC iteration (3N,).
    dt : float
        Time step (s).
    N : int
        Horizon length.
    inertia : np.ndarray
        3x3 inertia matrix.
    h_w : float
        Momentum wheel angular momentum.
    orbit_params : dict
        Orbital parameters.
    
    Returns
    -------
    x_traj : np.ndarray
        New reference trajectory (N+1, 7).
    B_traj_body : np.ndarray
        New magnetic field trajectory in body frame (N, 3).
    r_traj_body : np.ndarray
        New position vector trajectory in body frame (N, 3).
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params):
    def _get_rotation_matrix(q):
        q0, q1, q2, q3 = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])

    def _get_magnetic_field(t, orbit_params):
        R_e = 6371200.0
        M = 7.94e22
        mu0_over_4pi = 1e-7
        omega_e = 7.292115e-5
        tilt = np.radians(11.5)
        phi_m0 = np.radians(-70.0)
        
        r = R_e + orbit_params['altitude']
        inc = np.radians(orbit_params['inclination'])
        theta = orbit_params['omega_orbit'] * t
        r_eci = r * np.array([np.cos(theta), np.sin(theta)*np.cos(inc), np.sin(theta)*np.sin(inc)])
        
        phi_m = phi_m0 + omega_e * t
        m_eci = M * np.array([np.sin(tilt)*np.cos(phi_m), np.sin(tilt)*np.sin(phi_m), np.cos(tilt)])
        
        r_mag = np.linalg.norm(r_eci)
        r_hat = r_eci / r_mag
        return mu0_over_4pi * (1.0/r_mag**3) * (3.0*np.dot(m_eci, r_hat)*r_hat - m_eci)

    def _ode_fun(t, x, u, B_eci, r_eci):
        q = x[0:4]
        w = x[4:7]
        C = _get_rotation_matrix(q)
        B_body = C @ B_eci
        tau_mag = np.cross(u, B_body)
        
        mu_e = 3.986004418e14
        r_body = C @ r_eci
        r_norm = np.linalg.norm(r_body)
        tau_gg = (3.0 * mu_e / r_norm**5) * np.cross(r_body, np.dot(inertia, r_body))
        
        h_w_vec = np.array([0.0, 0.0, float(h_w)])
        H_total = np.dot(inertia, w) + h_w_vec
        w_dot = np.linalg.solve(inertia, (tau_mag + tau_gg) - np.cross(w, H_total))
        
        dqdt = 0.5 * np.array([
            -q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
             q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
             q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
             q[0]*w[2] + q[1]*w[1] - q[2]*w[0]
        ])
        return np.concatenate([dqdt, w_dot])

    x_traj = np.zeros((N + 1, 7))
    x_traj[0] = x0
    B_traj_body = np.zeros((N, 3))
    r_traj_body = np.zeros((N, 3))
    
    current_x = x0.copy()
    R_e = 6371200.0
    r_orbit = R_e + orbit_params['altitude']
    inc = np.radians(orbit_params['inclination'])
    
    for k in range(N):
        t_k = k * dt
        u_k = u_prev_seq[3*k : 3*k+3]
        
        B_eci = _get_magnetic_field(t_k, orbit_params)
        theta = orbit_params['omega_orbit'] * t_k
        r_eci = r_orbit * np.array([np.cos(theta), np.sin(theta)*np.cos(inc), np.sin(theta)*np.sin(inc)])
        
        C = _get_rotation_matrix(current_x[0:4])
        B_traj_body[k] = C @ B_eci
        r_traj_body[k] = C @ r_eci
        
        sol = solve_ivp(_ode_fun, [t_k, t_k + dt], current_x, args=(u_k, B_eci, r_eci), method='RK45')
        current_x = sol.y[:, -1]
        current_x[0:4] /= np.linalg.norm(current_x[0:4]) # Normalize quaternion
        x_traj[k + 1] = current_x
        
    return x_traj, B_traj_body, r_traj_body

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
x0 = np.array([1.0, 0, 0, 0, 0, 0, 0.1])
u_prev_seq = np.zeros(6)
dt = 0.5
N = 2
inertia = np.diag([0.1, 0.1, 0.05])
h_w = 0.01
orbit_params = {'altitude': 500000, 'inclination': 45.0, 'omega_orbit': 0.0011}
""",
            "call": "sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)",
            "gold_call": "_gold_sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)"
        },
        {
            "setup": """
x0 = np.array([1.0, 0, 0, 0, 0.01, 0.01, 0.1])
u_prev_seq = np.zeros(9)
dt = 1.0
N = 3
inertia = np.diag([0.03, 0.03, 0.015])
h_w = 0.05 # Paper value
orbit_params = {'altitude': 500000, 'inclination': 45.0, 'omega_orbit': 0.0011}
""",
            "call": "sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)",
            "gold_call": "_gold_sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)"
        },
        {
            "setup": """
x0 = np.array([0.9914, 0.1305, 0, 0, 0, 0, 0])
u_prev_seq = np.ones(30) * 0.05
dt = 2.0
N = 10
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.0
orbit_params = {'altitude': 400000, 'inclination': 90.0, 'omega_orbit': 0.0012}
""",
            "call": "sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)",
            "gold_call": "_gold_sl_mpc_update(x0, u_prev_seq, dt, N, inertia, h_w, orbit_params)"
        }
    ]

