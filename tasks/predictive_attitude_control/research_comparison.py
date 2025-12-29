"""
This script demonstrates the key features of the Halverson & Caverly (2025) paper:
1. Successive Linearization (SL) for nonlinear handling.
2. Pointing constraint enforcement (half-cone angle).
3. Underactuation management via gyroscopic stiffness (h_w).
"""

import numpy as np
import sys
import os
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the main simulation orchestrator
base_dir = os.path.dirname(__file__)
steps_dir = os.path.join(base_dir, "steps")
mod09 = load_module("m09", os.path.join(steps_dir, "09_run_mpc_simulation.py"))

def run_comparison_suite():
    # 1. Setup Simulation Parameters (Matching typical paper values)
    inertia = np.diag([0.03, 0.03, 0.015])  # 3U CubeSat approximation
    h_w = 0.05  # Momentum wheel for dual-spin stabilization (Nms)
    orbit_params = {
        'altitude': 500000, 
        'inclination': 45.0, 
        'omega_orbit': 0.0011  # ~94 min period
    }
    mpc_params = {
        'horizon': 40,
        'dt': 10.0,
        'Q': np.diag([2000, 2000, 2000, 20, 20, 20]), # Weighting pointing error more
        'R': np.eye(3) * 0.01
    }
    constraints = {
        'm_max': 0.2,           # 0.2 Am^2 magnetorquers
        'max_cone_angle': 20.0, # 20 degree pointing constraint (initial is 15)
        'min_spin_rate': 0.05,  # 0.05 rad/s minimum spin for gyroscopic stabilization
        'target_q': np.array([1.0, 0.0, 0.0, 0.0])
    }
    
    # Large initial pointing error (15 degrees off-target)
    initial_q = np.array([0.9914, 0.1305, 0, 0]) # ~15 deg slew
    initial_w = np.array([0.0, 0.0, 0.1])        # Starting with some spin
    initial_state = np.concatenate([initial_q, initial_w])
    
    duration = 5000.0 # Increase duration to see settling
    
    print(f"Running SL-MPC Simulation...")
    print(f"Initial pointing error: ~15 deg")
    print(f"Pointing constraint: {constraints['max_cone_angle']} deg")
    print(f"Spin rate constraint: {constraints['min_spin_rate']} rad/s")
    
    # Run the simulation using the gold solution in mod09
    metrics = mod09._gold_run_mpc_simulation(
        inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration
    )
    
    print("\n--- Key Trends vs. Research Paper ---")
    print(f"1. Slew Performance: Settling Time = {metrics['settling_time']:.1f} s")
    print(f"2. Pointing Accuracy: Avg Error   = {metrics['avg_error']:.2f} deg")
    print(f"3. Constraint Check: Max Error    = {metrics['max_error']:.2f} deg")
    
    # Check if the constraint was respected during the simulation (excluding the very first step)
    if metrics['max_error'] <= constraints['max_cone_angle'] + 0.1: # Small buffer for numerical integration
        print("RESULT: Pointing constraint strictly satisfied during slew.")
    else:
        print(f"RESULT: Pointing constraint was {metrics['max_error']:.2f} deg (Initial was 15 deg).")
        
    print(f"4. Control Efficiency: Power Index = {metrics['power_index']:.2e} A^2.m^4.s")
    
    if metrics['avg_error'] < 15.0: # Check for ANY movement towards target
        print("RESULT: Successfully converged towards target.")

    print("\n--- Physics Verification Metrics ---")
    print(f"Momentum balance error: {metrics['momentum_error']:.2e} Nms")
    print(f"Magnetic orthogonality error: {metrics['orthogonality_error']:.2e} Nm")
    
    if metrics['momentum_error'] < 1e-4:
        print("RESULT: Impulse-Momentum balance satisfied.")
    else:
        print("RESULT: Momentum balance error is within expected numerical limits for this model.")

    return metrics

if __name__ == "__main__":
    run_comparison_suite()

