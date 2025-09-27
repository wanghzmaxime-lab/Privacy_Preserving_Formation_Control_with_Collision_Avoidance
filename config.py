# config.py
# All simulation parameters and settings are defined here.

import numpy as np

# --- Simulation Parameters ---
SIMULATION_STEPS = 300   # Total number of steps to run the simulation
TIME_STEP = 0.05         # Duration of each simulation step

# --- Cryptography Parameters ---
KEY_LENGTH = 128             # Bit length for the Paillier keys
ENCODING_ACCURACY_Q = 100000 # Accuracy for NIPE encoding (denominator scale)

# --- Controller Parameters  ---
ALPHA = 0.8              # alfa
BETA_IJ = 2.0            # beta0
GAMMA = 20.0             # gamma
D = 1.0                  # D
KAPPA = 0.5              # Chatter attenuation decay factor
THETA = -0.8             # Chatter detection threshold

# --- Agent and Formation Definition (8 Agents) ---
NUM_AGENTS = 8
# Group velocity
GROUP_VELOCITY = np.array([1.0, 0.3])
# Desired formation shape
FORMATION_RADIUS = 5.0

# --- Initial States [pos_x, pos_y, vel_x, vel_y] ---
initial_positions = np.array([
    [ 0,  0], [ 5,  0], [10,  0], [15,  0],
    [ 0,  5], [ 5,  5], [10,  5], [15,  5]
])
initial_velocities = np.array([
    [ 1.2, -0.1], [-1.8,  0.1], [-0.3,  1.2], [ 1.2,  1.3],
    [-1.4,  0.5], [ 0.1, -1.6], [ 0.5, -0.2], [-0.3,  1.1]
])

INITIAL_STATES = np.hstack([initial_positions, initial_velocities]).tolist()

# --- Formation Targets [pos_x, pos_y, vel_x, vel_y] ---
angles = np.linspace(0, 2 * np.pi, NUM_AGENTS, endpoint=False)
target_positions = np.array([
    [FORMATION_RADIUS * np.cos(angle), FORMATION_RADIUS * np.sin(angle)] for angle in angles
])

FORMATION_TARGETS = [
    list(pos) + list(GROUP_VELOCITY) for pos in target_positions
]

# --- Communication Topology  ---
LAPLACIAN_MATRIX_L0 = np.array([
    [ 5,  0, -1,  0, -1, -1, -1, -1],
    [ 0,  3, -1, -1,  0,  0, -1,  0],
    [-1, -1,  3,  0,  0,  0, -1,  0],
    [ 0, -1,  0,  3,  0,  0, -1, -1],
    [-1,  0,  0,  0,  4, -1, -1, -1],
    [-1,  0,  0,  0, -1,  3,  0, -1],
    [-1, -1, -1, -1, -1,  0,  6, -1],
    [-1,  0,  0, -1, -1, -1, -1,  5]
])

ADJACENCY_MATRIX = (LAPLACIAN_MATRIX_L0 == -1).astype(int)

