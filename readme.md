## Simulation for "Privacy-Preserving Multi-Agent System Formation Control with Collision Avoidance"
### 1. Overview

This repository contains the Python simulation for the research paper titled "Privacy-Preserving Formation Control for Multi-Agent Systems with Collision Avoidance". The project aims to validate the proposed privacy-preserving formation control scheme, which is based on the Paillier homomorphic cryptosystem.

The simulation visualizes multiple agents operating in a 2D plane. While preserving the privacy of their initial states (position and velocity), the agents cooperatively achieve and maintain a desired formation, execute collective maneuvers, and effectively avoid collisions with one another.

### 2. Features

Dynamic Visualization Interface: Real-time display of agent trajectories, virtual node states, alert areas, and various performance metrics.

Privacy-Preserving Protocol Simulation: Implements the Paillier cryptosystem and the Privacy-Preserving Direction Determination (PPDD) protocol using the phe (Python Homomorphic Encryption) library.

Complete Control Law Implementation: The controller includes both a Formation Component and a Collision Avoidance Component to ensure stable convergence and operational safety.

Chatter Attenuation Mechanism: Implements a dynamic gain adjustment mechanism, as described in the paper, to produce smoother control inputs.

Multi-Plot Monitoring Dashboard:

    Real-time plot of the minimum distance between any two agents.

    Real-time plots of position and velocity error norms to quantify the consensus process.

    Live monitoring of the actual encrypted (E1, E2) and decrypted (D1, D2) data streams between key agents.

    Parameterized Configuration: All simulation parameters (agent count, initial states, communication topology, controller gains, etc.) are centralized in config.py for easy modification and testing.

### 3. Requirements

    Python 3.6+

    Third-party libraries:

        numpy

        matplotlib

        phe

### 4. Installation and Usage

Step 1: Clone or Download the Project

Step 2: Install Dependencies

Step 3: Run the Simulation

In the project's root directory, execute the main script:

python main.py

A dynamic visualization window will appear, displaying the entire formation simulation process.

### 5. Project Structure


├── agent.py               # Defines the Agent class's basic properties (state, keys, etc.)

├── config.py              # Stores all simulation parameters and initial settings

├── main.py                # The main entry point; initializes and runs the simulation loop

├── protocol.py            # Implements the control algorithms and cryptographic protocols

├── requirements.txt       # Lists the required Python libraries for the project

├── utils.py               # Provides helper functions

└── visualizer.py          # Manages the creation and updating of all visualization plots