# agent.py
# Defines the Agent class, now using the 'phe' library for cryptography.

import numpy as np
from phe import paillier
import config


class Agent:
    """
    Represents a single agent in the multi-agent system.
    Each agent generates its own Paillier keypair using the 'phe' library.
    """

    def __init__(self, agent_id, initial_state, formation_target):
        """
        Initializes an agent and its Paillier keypair.
        """
        self.id = agent_id
        self.position = np.array(initial_state[:2], dtype=float)
        self.velocity = np.array(initial_state[2:], dtype=float)

        self.formation_target_position = np.array(formation_target[:2], dtype=float)
        self.formation_target_velocity = np.array(formation_target[2:], dtype=float)

        self.control_input = np.zeros(2, dtype=float)

        self.p_tilde = np.zeros(2, dtype=float)
        self.v_tilde = np.zeros(2, dtype=float)
        self.s_tilde = np.zeros(2, dtype=float)

        # --- Generate Paillier keypair using the 'phe' library ---
        print(f"Generating Paillier keypair for Agent {self.id}...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=config.KEY_LENGTH)

        self.update_offset_states()
        print(f"Agent {self.id} created at {self.position}")

    def update_offset_states(self):
        """Calculates offset states based on current and target states."""
        self.p_tilde = self.position - self.formation_target_position
        self.v_tilde = self.velocity - self.formation_target_velocity
        self.s_tilde = config.ALPHA * self.p_tilde + self.v_tilde

    def update_state(self):
        """Updates the agent's physical state based on its control_input."""
        self.velocity += self.control_input * config.TIME_STEP
        self.position += self.velocity * config.TIME_STEP
        self.update_offset_states()

