# main.py
# Main entry point of the project
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agent import Agent
from visualizer import Visualizer
from utils import check_initial_conditions
import protocol
import config


def main():
    """
    Main function to initialize and run the simulation.
    """
    print("Starting privacy-preserving multi-agent formation simulation...")
    # Load parameters
    initial_states = config.INITIAL_STATES
    formation_targets = config.FORMATION_TARGETS
    adj_matrix = config.ADJACENCY_MATRIX
    num_agents = len(initial_states)

    # Check initial conditions
    if not check_initial_conditions(initial_states, config.D):
        print("Error: Initial conditions are not met. Some agents are too close.")
        return

    # Initialize agents
    print(f"Initializing {num_agents} agents...")
    agents = []
    for i in range(num_agents):
        agent = Agent(
            agent_id=i,
            initial_state=initial_states[i],
            formation_target=formation_targets[i]
        )
        agents.append(agent)
    print("Agents initialized.")

    # Set up visualization
    print("Setting up visualization window...")
    visualizer = Visualizer(agents)

    # --- Animation Setup using FuncAnimation ---
    print("Starting simulation loop...")

    # The update function for the animation. 'frame' is the step number.
    def update(frame):
        # Compute control input for all agents
        protocol.compute_all_control_inputs(agents, adj_matrix, frame)

        # Update state for each agent
        for agent in agents:
            agent.update_state()

        # Update the visualization and return the artists that have changed
        # This is crucial for blitting to work.
        print(f"Simulation Step: {frame}/{config.SIMULATION_STEPS}", end='\r')
        return visualizer.update(agents, frame)

    # Create the animation object
    # blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(
        fig=visualizer.fig,
        func=update,
        frames=config.SIMULATION_STEPS,
        interval=50,  # Delay between frames in milliseconds.
        blit=True,
        repeat=False  # Do not repeat the animation
    )

    plt.show()
    print("\nSimulation finished.")


if __name__ == "__main__":

    main()

