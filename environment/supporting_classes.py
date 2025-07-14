import numpy as np
import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import art3d  # Import for 3D patch objects

from environment.base_classes import Vector3D, Waypoint, EffectorKinematicState, EffectorWeaponState, ExplosiveType, ChassisType, DroneState

from environment.utils import calculate_spherical_coordinates, calculate_delta_azimuth_delta_elevation

class ScenarioRenderer:
    def __init__(self, domain_bb, sensitive_zones_bb, swarm_spawning_bb, swarm_intermediate_waypoints_bb,
                 effector_list, sensitive_zones, plot_trajectories=False, plot_detections=False, grid=False, render_mode="human"):
        self.render_mode = render_mode
        self.sensitive_zones = sensitive_zones

        if render_mode == "human":
            self.master = tk.Tk()
            self.master.title("Swarm Attack Scenario")
        else:
            # For rgb_array mode, we don't need a Tk window
            self.master = None

        self.bg_color = "black"
        self.base_line_color = "white"
        self.marker_size = 50

        self.plot_detections = plot_detections
        self.plot_trajectories = plot_trajectories

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.bg_color)
        self.fig.patch.set_facecolor(self.bg_color)

        # Set label colors
        self.ax.xaxis.label.set_color(self.base_line_color)
        self.ax.yaxis.label.set_color(self.base_line_color)
        self.ax.zaxis.label.set_color(self.base_line_color)

        # Set tick colors
        self.ax.tick_params(axis='x', colors=self.base_line_color)
        self.ax.tick_params(axis='y', colors=self.base_line_color)
        self.ax.tick_params(axis='z', colors=self.base_line_color)

        # Set color of the axes spines
        self.ax.xaxis.pane.set_edgecolor(self.base_line_color)
        self.ax.yaxis.pane.set_edgecolor(self.base_line_color)
        self.ax.zaxis.pane.set_edgecolor(self.base_line_color)

        # Set grid line colors
        gridcolor = self.base_line_color if grid == True else self.bg_color
        self.ax.xaxis._axinfo["grid"]['color'] = gridcolor
        self.ax.yaxis._axinfo["grid"]['color'] = gridcolor
        self.ax.zaxis._axinfo["grid"]['color'] = gridcolor

        # Set pane colors to black
        self.ax.xaxis.pane.set_facecolor(self.bg_color)
        self.ax.yaxis.pane.set_facecolor(self.bg_color)
        self.ax.zaxis.pane.set_facecolor(self.bg_color)

        # Draw domain BB
        self.ax.set_xlim(domain_bb.min.coords[0], domain_bb.max.coords[0])
        self.ax.set_ylim(domain_bb.min.coords[1], domain_bb.max.coords[1])
        self.ax.set_zlim(domain_bb.min.coords[2], domain_bb.max.coords[2])

        # Store the scatter plot objects
        self.drones_scatter_plot = []
        self.effectors_scatter_plot = []
        self.aiming_lines = []

        self.plot_cube(sensitive_zones_bb)
        self.plot_sensitive_zones(sensitive_zones)
        self.plot_cube(swarm_spawning_bb)
        #self.plot_cube(swarm_intermediate_waypoints_bb)
        plt.gca().set_aspect('equal')

        # Only create the Tk interface elements if in human render mode
        if self.render_mode == "human":
            # Create a PanedWindow to hold the matplotlib plot and the Treeviews
            self.paned_window = tk.PanedWindow(self.master, orient=tk.HORIZONTAL)
            self.paned_window.pack(fill=tk.BOTH, expand=True)

            # Create a frame to hold the matplotlib plot
            self.plot_frame = tk.Frame(self.paned_window)
            self.plot_frame.config(bg=self.bg_color)
            self.plot_frame.grid_rowconfigure(0, weight=1)
            self.plot_frame.grid_columnconfigure(0, weight=1)

            # Create a canvas for the matplotlib plot
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add the plot frame to the PanedWindow
            self.paned_window.add(self.plot_frame)

            self.create_tree_views_frame(sensitive_zones)

            # Add the tree frame to the PanedWindow
            self.paned_window.add(self.tree_frame)

            # Configure grid resizing behavior
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
        else:
            # For rgb_array mode, use a different canvas
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.canvas = FigureCanvasAgg(self.fig)

        self.update()

    def create_tree_views_frame(self, sensitive_zones):
        # Only create tree views if in human render mode
        if self.render_mode != "human":
            return

        # Create a frame to hold the Treeviews
        self.tree_frame = tk.Frame(self.paned_window)
        self.tree_frame.config(bg=self.bg_color)
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(1, weight=1)
        self.tree_frame.grid_rowconfigure(2, weight=10)
        self.tree_frame.grid_rowconfigure(3, weight=1)
        self.tree_frame.grid_rowconfigure(4, weight=10)
        self.tree_frame.grid_rowconfigure(5, weight=1)
        self.tree_frame.grid_rowconfigure(6, weight=10)
        self.tree_frame.grid_rowconfigure(7, weight=1)
        self.tree_frame.grid_rowconfigure(8, weight=2000)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        # Create a custom style for the Treeviews
        self.style = ttk.Style()
        self.style.theme_use("default")  # Use the default theme
        self.style.configure("Treeview",
                             background=self.bg_color,
                             foreground=self.base_line_color,
                             fieldbackground=self.bg_color)
        self.style.map("Treeview", background=[("selected", "blue")])

        # Stats
        # Title
        self.stats_tree_title = tk.Label(self.tree_frame, text="Stats", bg=self.bg_color, fg=self.base_line_color)
        self.stats_tree_title.grid(row=0, column=0, sticky="nsew")
        # Sub-Title 1
        self.stats_tree_subtitle_1 = tk.Label(self.tree_frame, text="Drones", bg=self.bg_color, fg=self.base_line_color)
        self.stats_tree_subtitle_1.grid(row=1, column=0, sticky="nsew")
        # Table
        self.drones_stats_tree = ttk.Treeview(self.tree_frame,columns=("Impacted", "Active", "Neutralized", "Damage [%]"), show="headings", height=1)
        self.drones_stats_tree.grid(row=2, column=0, sticky="nsew")
        # Add headers for drone stats
        for col in self.drones_stats_tree["columns"]:
            self.drones_stats_tree.heading(col, text=col, anchor="center")
            self.drones_stats_tree.column(col, anchor="center")

        # Sub-Title 2
        self.stats_tree_subtitle_2 = tk.Label(self.tree_frame, text="Impacted Sensitive Zones", bg=self.bg_color, fg=self.base_line_color)
        self.stats_tree_subtitle_2.grid(row=3, column=0, sticky="nsew")
        # Table
        columns = ("Base (Value: 1)",)
        for sensitive_zone in sensitive_zones:
            columns += (str(sensitive_zone.id) + " (Value: " + str(sensitive_zone.value) + ")",)
        self.zones_stats_tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=1)
        self.zones_stats_tree.grid(row=4, column=0, sticky="nsew")

        # Add headers for sensitive zones stats
        for col in self.zones_stats_tree["columns"]:
            self.zones_stats_tree.heading(col, text=col, anchor="center")
            self.zones_stats_tree.column(col, anchor="center")

        # Create a Treeview for the effectors (top half)
        # Title
        self.effectors_tree_title = tk.Label(self.tree_frame, text="Effectors", bg=self.bg_color, fg=self.base_line_color)
        self.effectors_tree_title.grid(row=5, column=0, sticky="nsew")
        # Table
        self.effectors_tree = ttk.Treeview(self.tree_frame,
                                           columns=("ID", "Position (x,y,z)", "Neutralization Count",
                                                    "Weapon State", "Kinematic State", "Tracking Steps [%]", "Chasing Steps [%]"),
                                           show="headings", height=4)
        self.effectors_tree.grid(row=6, column=0, sticky="nsew")
        # Scrollbar
        self.scrollbar_effectors = ttk.Scrollbar(self.tree_frame, orient="vertical")
        self.scrollbar_effectors.grid(row=6, column=1, sticky="ns")
        self.scrollbar_effectors.config(command=self.effectors_tree.yview)
        self.effectors_tree.configure(yscrollcommand=self.scrollbar_effectors.set)
        # Configure tag colors
        self.effectors_tree.tag_configure("READY", foreground="cyan")
        self.effectors_tree.tag_configure("SHOOTING", foreground="yellow")
        self.effectors_tree.tag_configure("RECHARGING", foreground="red")

        # Add headers for effectors
        for col in self.effectors_tree["columns"]:
            self.effectors_tree.heading(col, text=col, anchor="center")
            self.effectors_tree.column(col, anchor="center")

        self.effectors_tree.column("ID", width=40)

        # Create a Treeview for the drones (bottom half)
        # Title
        self.drones_tree_title = tk.Label(self.tree_frame, text="Drones", bg=self.bg_color, fg=self.base_line_color)
        self.drones_tree_title.grid(row=7, column=0, sticky="nsew")
        # Table
        self.drones_tree = ttk.Treeview(self.tree_frame,
                                        columns=("ID", "State", "Sensitive Target (ID - Value)", "Potential Damage",
                                                 "Chassis", "Explosive", "Detected Chassis", "Detected Explosive"),
                                        show="headings")
        self.drones_tree.grid(row=8, column=0, sticky="nsew")
        # Scrollbar
        self.scrollbar_drones = ttk.Scrollbar(self.tree_frame, orient="vertical")
        self.scrollbar_drones.grid(row=8, column=1, sticky="ns")
        self.scrollbar_drones.config(command=self.drones_tree.yview)
        self.drones_tree.configure(yscrollcommand=self.scrollbar_drones.set)
        # Configure tag colors
        self.drones_tree.tag_configure("IMPACTED", foreground="red")
        self.drones_tree.tag_configure("ACTIVE", foreground="yellow")
        self.drones_tree.tag_configure("NEUTRALIZED", foreground="cyan")

        # Add headers for bottom Treeview
        for col in self.drones_tree["columns"]:
            self.drones_tree.heading(col, text=col, anchor="center")
            self.drones_tree.column(col, anchor="center")

        # Set the width of each column in the bottom Treeview
        self.drones_tree.column("ID", width=40)
        self.drones_tree.column("Potential Damage", width=150)
        self.drones_tree.column("Chassis", width=100)
        self.drones_tree.column("Explosive", width=100)
        self.drones_tree.column("Detected Chassis", width=100)
        self.drones_tree.column("Detected Explosive", width=100)

    def plot_effectors_aiming(self, effector_list):
        for effector_scatter_plot in self.effectors_scatter_plot:
            effector_scatter_plot.remove()
        self.effectors_scatter_plot.clear()
        for aiming_line in self.aiming_lines:
            aiming_line[0].remove()
        self.aiming_lines.clear()

        for effector in effector_list:
            # Effector Marker
            effector_scatter_plot = self.ax.scatter([effector.location.coords[0]],
                                                    [effector.location.coords[1]],
                                                    [effector.location.coords[2]],
                                                    s=self.marker_size, marker=effector.marker, color=effector.color)
            self.effectors_scatter_plot.append(effector_scatter_plot)
            # Aiming Line
            x = [effector.location.coords[0], effector.location.coords[0] + effector.aiming_line_length * math.cos(effector.aiming[1])*math.cos(effector.aiming[0])]
            y = [effector.location.coords[1], effector.location.coords[1] + effector.aiming_line_length * math.cos(effector.aiming[1])*math.sin(effector.aiming[0])]
            z = [effector.location.coords[2], effector.location.coords[2] + effector.aiming_line_length * math.sin(effector.aiming[1])]
            aiming_line = self.ax.plot(x, y, z, linestyle=effector.aiming_line_style, color=effector.color)
            self.aiming_lines.append(aiming_line)

    def plot_sensitive_zones(self, sensitive_zones):
        for sensitive_zone in sensitive_zones:
            self.plot_circle(sensitive_zone.location.coords[0], sensitive_zone.location.coords[1], sensitive_zone.location.coords[2]+1,
                             sensitive_zone.radius, self.float_to_rgb(sensitive_zone.value))

    def plot_drones_trajectories(self, trajectories):
        if self.plot_trajectories:
            for trajectory in trajectories:
                x = [trajectory[i_waypoint].position.coords[0] for i_waypoint in range(len(trajectory))]
                y = [trajectory[i_waypoint].position.coords[1] for i_waypoint in range(len(trajectory))]
                z = [trajectory[i_waypoint].position.coords[2] for i_waypoint in range(len(trajectory))]

                self.ax.plot(x, y, z, linestyle='--', color=self.base_line_color)

    def plot_drones_positions(self, swarm_drones_list, tick):
        for drone_scatter_plot in self.drones_scatter_plot:
            drone_scatter_plot.remove()
        self.drones_scatter_plot.clear()

        # Add visualization enhancements for rgb_array mode
        if self.render_mode == "rgb_array":
            # Calculate statistics
            _, final_row, _ = self.calculate_drones_data(swarm_drones_list)
            impacted_count, active_count, neutralized_count, damage_percentage = final_row

            # Clear any existing text by removing all text objects
            for text in self.ax.texts:
                text.remove()

            # Add title at the top with nice background
            self.ax.text2D(0.5, 0.98, "Counter Drone Swarm Scenario",
                          transform=self.ax.transAxes,
                          fontsize=14, color='white',
                          horizontalalignment='center',
                          verticalalignment='top',
                          fontweight='bold')

            damage_text = f"Damage: {damage_percentage}%"
            self.ax.text2D(0.05, 0.05, damage_text,
                          transform=self.ax.transAxes,
                          fontsize=12, color="red",
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          bbox=dict(facecolor='#000000', alpha=0.8))

            # Impacted drones (red accent)
            self.ax.text2D(0.98, 0.14, f"Impacted: {impacted_count}",
                          transform=self.ax.transAxes,
                          fontsize=10, color='red',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          bbox=dict(facecolor='#000000', alpha=0.8))

            # Active drones (yellow accent)
            self.ax.text2D(0.98, 0.08, f"Active: {active_count}",
                          transform=self.ax.transAxes,
                          fontsize=10, color='red',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          bbox=dict(facecolor='#000000', alpha=0.8))

            # Neutralized drones (cyan accent)
            self.ax.text2D(0.98, 0.02, f"Neutralized: {neutralized_count}",
                          transform=self.ax.transAxes,
                          fontsize=10, color='red',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          bbox=dict(facecolor='#000000', alpha=0.8))

        for drone in swarm_drones_list:
            idx = min(tick, len(drone.trajectory)-1)
            drone_scatter_plot = self.ax.scatter([drone.trajectory[idx].position.coords[0]],
                                                 [drone.trajectory[idx].position.coords[1]],
                                                 [drone.trajectory[idx].position.coords[2]],
                                                 s=self.marker_size, marker=drone.marker, color=drone.marker_color)
            self.drones_scatter_plot.append(drone_scatter_plot)
            if self.plot_detections:
                drone_scatter_plot_detection = self.ax.scatter([drone.detections[idx].position.coords[0]],
                                                               [drone.detections[idx].position.coords[1]],
                                                               [drone.detections[idx].position.coords[2]],
                                                               s=self.marker_size, marker=drone.detections[idx].marker,
                                                               color=drone.detections[idx].marker_color)
                self.drones_scatter_plot.append(drone_scatter_plot_detection)

    def float_to_rgb(self, value):
        value = max(1, min(value, 10))
        position = (value - 1) / 9

        # Interpolate between yellow and red
        red = 1
        green = max(0, min(1, 1 - position))
        blue = 0

        return red, green, blue

    def plot_circle(self, center_x, center_y, altitude, radius, color):
        # Plot circles on the ground at z=0
        circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.5)
        self.ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=altitude, zdir="z")

    def plot_square(self, square_bb):
        # Draw a square at z=0
        square = plt.Rectangle((-5, -5), 10, 10, color=self.base_line_color, alpha=0.3)
        self.ax.add_patch(square)
        art3d.pathpatch_2d_to_3d(square, z=-10, zdir="z")

    def plot_cube(self, cube_bb):
        # Draw a cube
        cube = art3d.Poly3DCollection([
            [(cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2]), (cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2]),
             (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2]), (cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2])],  # Bottom face
            [(cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2]), (cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]),
             (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]), (cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2])],  # Top face
            [(cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2]), (cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2]),
             (cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]), (cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2])],  # Side faces
            [(cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2]), (cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2]),
             (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]), (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2])],
            [(cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2]), (cube_bb.min.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2]),
             (cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.max.coords[2]), (cube_bb.max.coords[0], cube_bb.min.coords[1], cube_bb.min.coords[2])],
            [(cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2]), (cube_bb.min.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]),
             (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.max.coords[2]), (cube_bb.max.coords[0], cube_bb.max.coords[1], cube_bb.min.coords[2])]
        ], color=self.base_line_color, alpha=0.0)
        self.ax.add_collection3d(cube)

    def calculate_drones_data(self, swarm_drones_list):
        drones_stats = {}
        zones_stats = {}
        actual_damage = 0
        potential_damage = 0
        rows = []

        for idx in range(len(self.sensitive_zones) + 1):
            zones_stats[idx] = 0

        # ("ID", "State", "Chassis", "Explosive", "Detected Chassis", "Detected Explosive")
        for state in [DroneState.IMPACTED, DroneState.ACTIVE, DroneState.NEUTRALIZED]:
            drones_stats[state] = 0
            for idx, drone in enumerate(swarm_drones_list):
                if drone.state == state:
                    drones_stats[state] += 1
                    if drone.sensitive_zone_targeted is not None:
                        sensitive_zone = "ID: " + str(drone.sensitive_zone_targeted.id) + " - Value: " + str(drone.sensitive_zone_targeted.value)
                        if state == DroneState.IMPACTED:
                            zones_stats[drone.sensitive_zone_targeted.id + 1] += 1
                    else:
                        sensitive_zone = "ID: N/A - Value: 1"
                        if state == DroneState.IMPACTED:
                            zones_stats[0] += 1
                    if state == DroneState.IMPACTED:
                        actual_damage += drone.potential_damage
                    potential_damage += drone.potential_damage
                    row=(idx, drone.state.name, sensitive_zone, drone.potential_damage, drone.chassis.name,
                        drone.explosive.name, drone.detections[-1].chassis.name, drone.detections[-1].explosive.name)
                    rows.append(row)

        final_row=(drones_stats[DroneState.IMPACTED], drones_stats[DroneState.ACTIVE],
             drones_stats[DroneState.NEUTRALIZED], round(actual_damage/potential_damage * 100, 2))

        return rows, final_row, zones_stats

    def update_drones_data(self, swarm_drones_list):
        # Skip updating tree views in rgb_array mode
        if self.render_mode != "human":
            return

        for item in self.drones_tree.get_children():
            self.drones_tree.delete(item)
        for item in self.drones_stats_tree.get_children():
            self.drones_stats_tree.delete(item)
        for item in self.zones_stats_tree.get_children():
            self.zones_stats_tree.delete(item)

        rows, final_row, zones_stats = self.calculate_drones_data(swarm_drones_list)
        for row in rows:
            self.drones_tree.insert("", "end", values=row, tags=(row[1],))

        self.drones_stats_tree.insert("", "end", values=final_row)

        row = ()
        for idx in range(len(self.zones_stats_tree["columns"])):
            row += (zones_stats[idx], )
        self.zones_stats_tree.insert("", "end", values=row)

    def update_effectors_data(self, effectors_list):
        # Skip updating tree views in rgb_array mode
        if self.render_mode != "human":
            return

        # Clear the Treeview
        for item in self.effectors_tree.get_children():
            self.effectors_tree.delete(item)

        # ("ID", "Position", "Weapon State", "Kinematic State", "Neutralization Count")
        for idx, effector in enumerate(effectors_list):
            position = str(effector.location.coords[0]) + ", " + str(effector.location.coords[1]) + ", " + str(effector.location.coords[2])
            total_steps = max(1, effector.kinematic_states_counts[EffectorKinematicState.TRACKING] + effector.kinematic_states_counts[EffectorKinematicState.CHASING])
            row=(idx, position, effector.neutralized, effector.weapon_state.name, effector.kinematic_state.name,
                 round(effector.kinematic_states_counts[EffectorKinematicState.TRACKING]/total_steps * 100, 2),
                 round(effector.kinematic_states_counts[EffectorKinematicState.CHASING]/total_steps * 100, 2))
            self.effectors_tree.insert("", "end", values=row, tags=(effector.weapon_state.name,))

    def update(self):
        # Redraw the plot
        self.canvas.draw()

        # Only update Tk if in human render mode
        if self.render_mode == "human" and self.master is not None:
            self.master.update()

    def get_rgb_array(self):
        # Draw the figure to a canvas
        self.canvas.draw()

        # Get the RGB buffer from the figure - for FigureCanvasAgg
        buf = np.array(self.canvas.renderer.buffer_rgba())
        # Convert RGBA to RGB
        buf = buf[:, :, :3]

        # Return the RGB array
        return buf

class SensitiveZone:
    id: int = None
    location: Vector3D = None
    radius: float = None
    value: float = None
    impacts: int = 0

    def __init__(self, id, location, radius, value):
        self.id = id
        self.location = location
        self.radius = radius
        self.value = value

    def random_inner_point(self):
        theta = self.np_random.uniform(0, 2 * np.pi)
        r = np.sqrt(self.np_random.uniform(0, 1)) * self.radius

        x = self.location.coords[0] + r * np.cos(theta)
        y = self.location.coords[1] + r * np.sin(theta)
        z = 0
        return Vector3D(x, y, z)

    def contains(self, position):
        distance = np.linalg.norm(position[:2] - self.location.coords[:2])
        return position[2] == self.location.coords[2] and distance < self.radius

    def intersects(self, zone):
        centers_distance = np.linalg.norm(zone.location.coords[:2] - self.location.coords[:2])
        sum_radii = zone.radius + self.radius
        return centers_distance < sum_radii

    def reset(self, np_random):
        self.np_random = np_random
        self.impacts = 0

class Effector:
    id: str = None
    location: Vector3D = None
    aiming: list[float, float] = [0, 0] # Azimuth - Elevation
    shooting_time: float = None
    max_angular_speeds: list[float, float] = None # Azimuth - Elevation
    kinematic_state: EffectorKinematicState = EffectorKinematicState.CHASING
    weapon_state: EffectorWeaponState = EffectorWeaponState.READY
    kinematic_states_counts: dict = {
        EffectorKinematicState.CHASING: 0,
        EffectorKinematicState.TRACKING: 0,
    }
    neutralized: int
    time_step: float = None
    marker = "D"
    color_map = {
        EffectorWeaponState.READY: "cyan",
        EffectorWeaponState.SHOOTING: "magenta",
        EffectorWeaponState.RECHARGING: "yellow",
    }
    aiming_line_length = 225
    aiming_line_style_map = {
        EffectorKinematicState.CHASING: "--",
        EffectorKinematicState.TRACKING: "-"
    }

    def __init__(self, id, location, shooting_time, max_angular_speeds):
        self.id = id
        self.location = location
        self.shooting_time = shooting_time
        self.max_angular_speeds = [speed * math.pi for speed in max_angular_speeds]
        self._set_rendering_style()

    def reset(self, time_step):
        self.aiming = [0, 0] # Azimuth - Elevation
        self.kinematic_state = EffectorKinematicState.CHASING
        self.weapon_state = EffectorWeaponState.READY
        self.time_step = time_step
        self.neutralized = 0
        self.shooting_ticks = 0
        self.max_shooting_ticks = int((self.shooting_time / self.time_step) + 0.5)
        self._set_rendering_style()
        self.kinematic_states_counts = {
            EffectorKinematicState.CHASING: 0,
            EffectorKinematicState.TRACKING: 0,
        }

    def assign_target(self, target_position: Vector3D):
        azimuth, elevation, _, distance_xy = calculate_spherical_coordinates(target_position.coords, self.location.coords)
        if distance_xy < 0.01:
            azimuth = self.aiming[0]

        # Limit Az-El by max angular speeds
        delta_azimuth, delta_elevation = calculate_delta_azimuth_delta_elevation(azimuth, elevation, self.aiming[0], self.aiming[1])
        min_reachable_delta_az = min(self.max_angular_speeds[0] * self.time_step, abs(delta_azimuth))

        self.aiming[0] += np.sign(delta_azimuth) * min_reachable_delta_az
        if self.aiming[0] > math.pi:
            self.aiming[0] -= 2*math.pi
        elif self.aiming[0] < -math.pi:
            self.aiming[0] += 2*math.pi

        self.aiming[1] += np.sign(delta_elevation) * min(abs(delta_elevation), self.max_angular_speeds[1] * self.time_step)

        if (self.aiming[0] == azimuth and self.aiming[1] == elevation):
            self.kinematic_state = EffectorKinematicState.TRACKING
        else:
            self.kinematic_state = EffectorKinematicState.CHASING

        if self.weapon_state == EffectorWeaponState.READY:
            if self.kinematic_state == EffectorKinematicState.TRACKING:
                self.weapon_state = EffectorWeaponState.SHOOTING
        elif self.weapon_state == EffectorWeaponState.SHOOTING:
            self.weapon_state = EffectorWeaponState.RECHARGING

        if self.weapon_state != EffectorWeaponState.READY:
            self.shooting_ticks +=1

        if self.shooting_ticks == self.max_shooting_ticks:
            self.shooting_ticks = 0
            self.weapon_state = EffectorWeaponState.READY

        self._set_rendering_style()
        self.kinematic_states_counts[self.kinematic_state] += 1

    def _set_rendering_style(self):
        self.color = self.color_map[self.weapon_state]
        self.aiming_line_style = self.aiming_line_style_map[self.kinematic_state]

class Detection:
    position: Vector3D = None
    position_uncertainty: float = None
    velocity: Vector3D = None
    velocity_uncertainty: float = None
    explosive: ExplosiveType = None
    chassis: ChassisType = None

    def __init__(self, position, position_uncertainty, chassis, explosive):
        self.position = position
        self.position_uncertainty = position_uncertainty
        self.chassis = chassis
        self.explosive = explosive
        self.marker_color = {ExplosiveType.LIGHT: "yellow", ExplosiveType.MEDIUM: "orange", ExplosiveType.STRONG: "red"}[self.explosive]
        self.marker = {ChassisType.LARGE: "p", ChassisType.MEDIUM: "s", ChassisType.SMALL: "^"}[self.chassis]

class Drone:
    max_speed: float = None
    explosive: ExplosiveType = None
    chassis: ChassisType = None
    marker_color: str = 'white'
    marker: str = ""
    trajectory: list[Waypoint] = None
    potential_damage: float = 0.0
    detections: list[Detection] = None
    state: DroneState = DroneState.ACTIVE
    sensitive_zone_targeted: SensitiveZone = None

    def __init__(self, max_speed, explosive, chassis):
        self.max_speed = max_speed
        self.explosive = explosive
        self.chassis = chassis
        self.marker_color = {ExplosiveType.LIGHT: "yellow", ExplosiveType.MEDIUM: "orange", ExplosiveType.STRONG: "red"}[self.explosive]
        self.marker = {ChassisType.LARGE: "p", ChassisType.MEDIUM: "s", ChassisType.SMALL: "^"}[self.chassis]

    def neutralize(self, tick):
        self.state = DroneState.NEUTRALIZED
        for idx in range(tick+1, len(self.trajectory)):
            self.trajectory[idx].position.coords = self.trajectory[tick].position.coords
            self.marker_color = "cyan"
            self.marker = "x"
            self.detections[idx].position.coords = self.detections[tick].position.coords
            self.detections[idx].marker_color = "cyan"
            self.detections[idx].marker = "x"

    def _render_frame(self):
        self.renderer.plot_drones_positions(self.swarm_drones_list, self.tick)
        self.renderer.plot_effectors_aiming(self.effectors_list)
        self.renderer.update_drones_data(self.swarm_drones_list)
        self.renderer.update_effectors_data(self.effectors_list)

        if self.render_mode == "human":
            self.renderer.update()
        elif self.render_mode == "rgb_array":
            return self.renderer.get_rgb_array()