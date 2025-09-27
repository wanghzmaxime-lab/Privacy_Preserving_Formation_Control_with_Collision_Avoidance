# protocol.py
# Implements the control logic and privacy-preserving protocols using 'phe'.

import numpy as np
import random
import config

# --- Global state for Chatter Attenuation and Data Logging ---
protocol_state = {}


def initialize_protocol_state(num_agents):
    """Initializes the state for the protocol mechanisms."""
    global protocol_state
    protocol_state = {
        'dua_history_k1': np.zeros((num_agents, num_agents, 2)),
        'dua_history_k2': np.zeros((num_agents, num_agents, 2)),
        'beta_matrix': np.full((num_agents, num_agents), config.BETA_IJ),
        'c_counter': np.zeros((num_agents, num_agents), dtype=int),
        'latest_E1': None,
        'latest_E2': None,
        'latest_D1': None,
        'latest_D2': None,
    }


def secure_PPDD(agent_A, vector_a, agent_B, vector_b, interaction_type=None):
    """
    Simulates the full Privacy-Preserving Direction Determination (Protocol 1).
    This version implements a number encoding scheme that produces very large
    decrypted integers (D1, D2), matching the MATLAB reference implementation.
    """
    pub_key_A = agent_A.public_key
    priv_key_A = agent_A.private_key
    Q = config.ENCODING_ACCURACY_Q
    Q_squared = Q ** 2

    # --- Number Encoding (formerly NIPE) ---
    # The logic is now integrated here to create common denominators for each vector,
    # which is crucial for the math to work correctly. A large random scaler
    # is used to achieve the desired 10^36 magnitude for D1/D2.

    # Step 1: Party A encodes and encrypts
    # Generate one large, random common denominator for vector 'a'
    a_q = random.randint(Q, Q_squared)
    a_p1 = int(round(vector_a[0] * a_q))
    a_p2 = int(round(vector_a[1] * a_q))

    enc_ap1 = pub_key_A.encrypt(a_p1)
    enc_ap2 = pub_key_A.encrypt(a_p2)
    enc_neg_aq = pub_key_A.encrypt(-a_q)

    # Step 2: Party B encodes and performs homomorphic operations
    # Generate one large, random common denominator for vector 'b'
    b_q = random.randint(Q, Q_squared)
    b_p1 = int(round(vector_b[0] * b_q))
    b_p2 = int(round(vector_b[1] * b_q))

    # Use a large random integer for the blinding factor, as specified.
    r_b = random.randint(1, 2 ** 53 - 1)

    sigma = 0.1  # As per MATLAB code for rb1, rb2
    # Ensure r_b1, r_b2 are small integers
    r_b1 = random.randint(int(-sigma * (r_b - 1)), int(sigma * (r_b - 1))) if r_b > 1 else 0
    r_b2 = random.randint(int(-sigma * (r_b - 1)), int(sigma * (r_b - 1))) if r_b > 1 else 0

    E1 = (enc_ap1 * (b_q * r_b)) + (enc_neg_aq * (b_p1 * r_b)) + pub_key_A.encrypt(r_b1)
    E2 = (enc_ap2 * (b_q * r_b)) + (enc_neg_aq * (b_p2 * r_b)) + pub_key_A.encrypt(r_b2)

    # Step 3: Party A decrypts
    D1 = priv_key_A.decrypt(E1)
    D2 = priv_key_A.decrypt(E2)

    # Data Logging for Visualization
    agent_ids = {agent_A.id, agent_B.id}
    if agent_ids == {0, 4} and interaction_type == 's_tilde':
        protocol_state['latest_E1'] = E1.ciphertext(be_secure=False)
        protocol_state['latest_E2'] = E2.ciphertext(be_secure=False)
        protocol_state['latest_D1'] = D1
        protocol_state['latest_D2'] = D2

    # Convert the very large integers D1 and D2 to floats before using them in numpy.
    D1_float = float(D1)
    D2_float = float(D2)
    norm = np.sqrt(D1_float ** 2 + D2_float ** 2)

    if norm < 1e-9:
        return np.zeros(2)

    return np.array([D1_float / norm, D2_float / norm])


def compute_all_control_inputs(agents, adj_matrix, step):
    """
    Computes control inputs for all agents using the secure PPDD protocol.
    """
    num_agents = len(agents)
    if step == 0:
        initialize_protocol_state(num_agents)

    global protocol_state
    protocol_state['latest_E1'], protocol_state['latest_E2'] = None, None
    protocol_state['latest_D1'], protocol_state['latest_D2'] = None, None

    new_control_inputs = {}

    for i, agent_i in enumerate(agents):
        collision_neighbors_ids = set()
        u_collision_avoidance = np.zeros(2)

        for j, agent_j in enumerate(agents):
            if i == j: continue
            if np.linalg.norm(agent_j.position - agent_i.position, ord=np.inf) < config.D:
                collision_neighbors_ids.add(j)
                unit_vector_p = secure_PPDD(agent_j, agent_j.position, agent_i, agent_i.position,
                                            interaction_type='position')
                u_collision_avoidance -= config.GAMMA * unit_vector_p

        for neighbor_id in range(num_agents):
            if i == neighbor_id: continue
            is_colliding = neighbor_id in collision_neighbors_ids
            is_neighbor = adj_matrix[i, neighbor_id] == 1
            if is_colliding or not is_neighbor:
                protocol_state['c_counter'][i, neighbor_id] = 0
                protocol_state['beta_matrix'][i, neighbor_id] = config.BETA_IJ

        u_formation = -config.ALPHA * agent_i.v_tilde
        sum_term_s = np.zeros(2)
        communication_neighbors = np.where(adj_matrix[i] == 1)[0]

        for j in communication_neighbors:
            if j in collision_neighbors_ids:
                continue

            agent_j = agents[j]
            dua_current = secure_PPDD(agent_j, agent_j.s_tilde, agent_i, agent_i.s_tilde, interaction_type='s_tilde')

            if step > 1:
                dua_k1 = protocol_state['dua_history_k1'][i, j]
                dua_k2 = protocol_state['dua_history_k2'][i, j]
                du_k = dua_current - dua_k1
                du_k1 = dua_k1 - dua_k2
                if np.dot(du_k, du_k1) < config.THETA:
                    protocol_state['c_counter'][i, j] += 1

            beta_ij = config.BETA_IJ * (config.KAPPA ** protocol_state['c_counter'][i, j])
            protocol_state['beta_matrix'][i, j] = beta_ij
            sum_term_s += beta_ij * dua_current

            protocol_state['dua_history_k2'][i, j] = protocol_state['dua_history_k1'][i, j]
            protocol_state['dua_history_k1'][i, j] = dua_current

        u_formation += sum_term_s
        new_control_inputs[i] = u_formation + u_collision_avoidance

    for i, agent in enumerate(agents):
        agent.control_input = new_control_inputs[i]

