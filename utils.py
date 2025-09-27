# utils.py
# Contains helper functions.

import numpy as np


def check_initial_conditions(initial_states, D):
    """
    Checks if the initial conditions are valid.
    According to Assumption 1.3: For any two agents i, j, the initial positions must satisfy ||p_j[0] - p_i[0]||_inf >= D.
    :param initial_states: List of initial states for all agents.
    :param D: Size of the alert area.
    :return: True if conditions are met, False otherwise.
    """
    num_agents = len(initial_states)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            pos_i = np.array(initial_states[i][:2])
            pos_j = np.array(initial_states[j][:2])

            # Calculate the infinity norm distance
            dist_inf = np.linalg.norm(pos_j - pos_i, np.inf)

            if dist_inf < D:
                print(
                    f"Initial condition check failed: Distance between Agent {i} and Agent {j} is too small ({dist_inf} < {D})")
                return False

    print("Initial conditions check passed.")
    return True

