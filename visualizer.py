# visualizer.py
# Responsible for graphically displaying the simulation process.
# This version is optimized for use with matplotlib's FuncAnimation.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import config
import protocol


class Visualizer:
    """
    Visualization class with a multi-plot dashboard layout, optimized for FuncAnimation.
    """

    def __init__(self, agents):
        """
        Initializes the visualization window.
        :param agents: List of agent objects.
        """
        self.fig = plt.figure(figsize=(16, 9))

        gs_main = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[3, 1])
        gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1])

        self.ax_main = self.fig.add_subplot(gs_left[0])
        self.ax_formation = self.fig.add_subplot(gs_right[0])
        self.ax_decrypted = self.fig.add_subplot(gs_right[1])
        self.ax_encrypted = self.fig.add_subplot(gs_right[2])

        gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_left[1])
        self.ax_min_dist = self.fig.add_subplot(gs_bottom_left[0, 0])
        self.ax_pos_error = self.fig.add_subplot(gs_bottom_left[0, 1])
        self.ax_vel_error = self.fig.add_subplot(gs_bottom_left[0, 2])

        # --- Configure Main Plot ---
        self.ax_main.set_aspect('equal', adjustable='box')
        self.ax_main.grid(True)
        self.ax_main.set_title("Agent Trajectories, Virtual Nodes, and Alert Areas")
        self.ax_main.set_xlabel("X-axis")
        self.ax_main.set_ylabel("Y-axis")
        self.ax_main.set_xlim(-2.5, 40)
        self.ax_main.set_ylim(-2.5, 20)

        for ax in [self.ax_decrypted, self.ax_encrypted, self.ax_min_dist, self.ax_pos_error, self.ax_vel_error]:
            ax.tick_params(axis='x', labelbottom=False)
        # --- Configure Subplots ---
        self.ax_formation.set_title("Expected Formation", fontsize='small')
        self.ax_formation.set_aspect('equal', adjustable='box')
        self.ax_formation.axis('off')

        self.ax_decrypted.set_title("Decrypted Data D1, D2 (between A1, A5)", fontsize='small')
        self.ax_decrypted.grid(True)
        self.ax_encrypted.set_title("Encrypted Data E1, E2 (between A1, A5)", fontsize='small')
        self.ax_encrypted.grid(True)
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        self.ax_encrypted.yaxis.set_major_formatter(formatter)

        self.ax_min_dist.set_title("Min. Distance", fontsize='small')
        self.ax_min_dist.grid(True)
        self.ax_pos_error.set_title("Position Error", fontsize='small')
        self.ax_pos_error.grid(True)
        self.ax_vel_error.set_title("Velocity Error", fontsize='small')
        self.ax_vel_error.grid(True)

        # --- Initialize Plotting Elements ---
        num_agents = len(agents)
        self.colors = plt.cm.jet(np.linspace(0, 1, num_agents))
        self.agent_plots, self.trajectory_plots, self.s_tilde_plots, self.alert_area_patches = [], [], [], []
        self.trajectories = [[] for _ in range(num_agents)]
        self.time_history, self.min_dist_history = [], []
        self.pos_error_history = [[] for _ in range(num_agents)]
        self.vel_error_history = [[] for _ in range(num_agents)]
        self.e1_history, self.e2_history, self.d1_history, self.d2_history = [], [], [], []

        for i in range(num_agents):
            self.agent_plots.append(self.ax_main.plot([], [], 'o', color=self.colors[i], markersize=4)[0])
            self.trajectory_plots.append(self.ax_main.plot([], [], '-', color=self.colors[i], alpha=0.5)[0])
            self.s_tilde_plots.append(self.ax_main.plot([], [], 'p', color=self.colors[i], markersize=4)[0])
            rect = patches.Rectangle((0, 0), 2 * config.D, 2 * config.D, lw=1, ec='r', fc='r', alpha=0.15)
            self.alert_area_patches.append(rect)
            self.ax_main.add_patch(rect)
            target_pos = np.array(config.FORMATION_TARGETS[i][:2])
            self.ax_formation.plot(target_pos[0], target_pos[1], '*', color=self.colors[i], markersize=8)
            # --- ADDED: Agent labels on the formation plot ---
            self.ax_formation.text(target_pos[0] + 0.5, target_pos[1], f'A{i+1}',
                                   color='black', fontsize='small', ha='left', va='center')


        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if config.ADJACENCY_MATRIX[i, j] == 1:
                    pos_i, pos_j = np.array(config.FORMATION_TARGETS[i][:2]), np.array(config.FORMATION_TARGETS[j][:2])
                    self.ax_formation.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', alpha=0.3, lw=1)

        self.pos_error_plots = [self.ax_pos_error.plot([], [], '-', color=self.colors[i], lw=1)[0] for i in
                                range(num_agents)]
        self.vel_error_plots = [self.ax_vel_error.plot([], [], '-', color=self.colors[i], lw=1)[0] for i in
                                range(num_agents)]
        self.min_dist_plot, = self.ax_min_dist.plot([], [], '-k', lw=1.5)
        self.d1_plot, = self.ax_decrypted.plot([], [], '-', label='D1')
        self.d2_plot, = self.ax_decrypted.plot([], [], '-', label='D2')
        self.e1_plot, = self.ax_encrypted.plot([], [], '-', label='E1')
        self.e2_plot, = self.ax_encrypted.plot([], [], '-', label='E2')
        self.ax_decrypted.legend(fontsize='x-small')
        self.ax_encrypted.legend(fontsize='x-small')

        # Using full keyword arguments for Line2D
        agent_handles = [plt.Line2D([0], [0], color=self.colors[i], marker='o', linestyle='-',
                                    markersize=4, label=f'A{i+1}') for i in range(num_agents)]
        s_tilde_handles = [plt.Line2D([0], [0], color=self.colors[i], marker='p', linestyle='None',
                                      markersize=4, label=f's_tilde{i+1}') for i in range(num_agents)]
        self.ax_main.legend(handles=agent_handles + s_tilde_handles, loc='lower right', ncol=2, fontsize='x-small')

        self.fig.tight_layout()

        self.all_artists = self.agent_plots + self.trajectory_plots + self.s_tilde_plots + \
                           self.alert_area_patches + self.pos_error_plots + self.vel_error_plots + \
                           [self.min_dist_plot, self.d1_plot, self.d2_plot, self.e1_plot, self.e2_plot]

        # --- ADDED SECTION: Position the window ---
        try:
            # This works for many backends like TkAgg to position the window
            mngr = plt.get_current_fig_manager()
            # The geometry string "+0-0" is a common way to specify bottom-left
            mngr.window.wm_geometry("+0+2")
        except Exception:
            # If the backend does not support this, the window will appear
            # at the default position, which is an acceptable fallback.
            pass

    def update(self, agents, current_step):
        """
        Update function for FuncAnimation.
        """
        current_time = current_step * config.TIME_STEP
        self.time_history.append(current_time)

        all_positions = np.array([agent.position for agent in agents])
        centroid = np.mean(all_positions, axis=0)

        mean_p_tilde = np.mean([agent.p_tilde for agent in agents], axis=0)
        mean_v_tilde = np.mean([agent.v_tilde for agent in agents], axis=0)

        min_dist = float('inf')
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = np.linalg.norm(agents[i].position - agents[j].position)
                min_dist = min(min_dist, dist)
        self.min_dist_history.append(min_dist)

        for i, agent in enumerate(agents):
            self.agent_plots[i].set_data([agent.position[0]], [agent.position[1]])
            self.trajectories[i].append(agent.position.copy())
            traj_data = np.array(self.trajectories[i])
            self.trajectory_plots[i].set_data(traj_data[:, 0], traj_data[:, 1])
            s_tilde_display_pos = centroid + agent.s_tilde
            self.s_tilde_plots[i].set_data([s_tilde_display_pos[0]], [s_tilde_display_pos[1]])
            self.alert_area_patches[i].set_xy((agent.position[0] - config.D, agent.position[1] - config.D))

            pos_error = np.linalg.norm(agent.p_tilde - mean_p_tilde)
            self.pos_error_history[i].append(pos_error)
            self.pos_error_plots[i].set_data(self.time_history, self.pos_error_history[i])

            vel_error = np.linalg.norm(agent.v_tilde - mean_v_tilde)
            self.vel_error_history[i].append(vel_error)
            self.vel_error_plots[i].set_data(self.time_history, self.vel_error_history[i])

        self.min_dist_plot.set_data(self.time_history, self.min_dist_history)

        d1 = protocol.protocol_state.get('latest_D1')
        d2 = protocol.protocol_state.get('latest_D2')
        e1 = protocol.protocol_state.get('latest_E1')
        e2 = protocol.protocol_state.get('latest_E2')
        self.d1_history.append(d1 if d1 is not None else np.nan)
        self.d2_history.append(d2 if d2 is not None else np.nan)
        self.e1_history.append(e1 if e1 is not None else np.nan)
        self.e2_history.append(e2 if e2 is not None else np.nan)

        self.d1_plot.set_data(self.time_history, self.d1_history)
        self.d2_plot.set_data(self.time_history, self.d2_history)
        self.e1_plot.set_data(self.time_history, self.e1_history)
        self.e2_plot.set_data(self.time_history, self.e2_history)

        for ax in [self.ax_min_dist, self.ax_pos_error, self.ax_vel_error, self.ax_decrypted, self.ax_encrypted]:
            ax.relim()
            ax.autoscale_view()

        return self.all_artists

