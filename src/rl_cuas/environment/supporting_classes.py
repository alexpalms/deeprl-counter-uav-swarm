"""Supporting classes for the CUAS domain."""

import math
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d  # pyright:ignore[reportMissingTypeStubs]

from rl_cuas.environment.base_classes import (
    BoundingBox,
    ChassisType,
    DroneState,
    EffectorKinematicState,
    EffectorWeaponState,
    ExplosiveType,
    Vector3D,
    Waypoint,
)
from rl_cuas.environment.utils import (
    calculate_delta_azimuth_delta_elevation,
    calculate_spherical_coordinates,
)


class SensitiveZone:
    """SensitiveZone class.

    Parameters
    ----------
    id: int
        The id of the sensitive zone.
    """

    id: int
    location: Vector3D
    radius: float = 0.0
    value: float = 0.0
    impacts: int = 0

    def __init__(
        self, identifier: int, location: Vector3D, radius: float, value: float
    ):
        self.id = identifier
        self.location = location
        self.radius = radius
        self.value = value

    def random_inner_point(self) -> Vector3D:
        """
        Generate a random inner point of the sensitive zone.

        Returns
        -------
        point: Vector3D
            The random inner point.
        """
        theta = self.np_random.uniform(0, 2 * np.pi)
        r = np.sqrt(self.np_random.uniform(0, 1)) * self.radius

        x = self.location.coords[0] + r * np.cos(theta)
        y = self.location.coords[1] + r * np.sin(theta)
        z = 0
        return Vector3D(x, y, z)

    def contains(self, position: Vector3D) -> bool:
        """
        Check if a position is inside the sensitive zone.

        Parameters
        ----------
        position: Vector3D
            The position to check.

        Returns
        -------
        contains: bool
            True if the position is inside the sensitive zone, False otherwise.
        """
        distance = np.linalg.norm(position.coords[:2] - self.location.coords[:2])
        return bool(
            position.coords[2] == self.location.coords[2] and distance < self.radius
        )

    def intersects(self, zone: "SensitiveZone") -> bool:
        """
        Check if a sensitive zone intersects with another sensitive zone.

        Parameters
        ----------
        zone: SensitiveZone
            The sensitive zone to check intersection with.
        """
        centers_distance = np.linalg.norm(
            zone.location.coords[:2] - self.location.coords[:2]
        )
        sum_radii = zone.radius + self.radius
        return bool(centers_distance < sum_radii)

    def reset(self, np_random: np.random.Generator) -> None:
        """
        Reset the sensitive zone.

        Parameters
        ----------
        np_random: np.random.Generator
            The random number generator.
        """
        self.np_random = np_random
        self.impacts = 0


class Effector:
    """
    Effector class.

    Parameters
    ----------
    id: str
        The id of the effector.
    location: Vector3D
        The location of the effector.
    shooting_time: float
        The shooting time of the effector.
    max_angular_speeds: list[float]
        The max angular speeds of the effector.
    """

    identifier: str = ""
    location: Vector3D
    aiming: list[float] = [0.0, 0.0]  # Azimuth - Elevation
    shooting_time: float
    max_angular_speeds: list[float] = [0.0, 0.0]  # Azimuth - Elevation
    kinematic_state: EffectorKinematicState = EffectorKinematicState.CHASING
    weapon_state: EffectorWeaponState = EffectorWeaponState.READY
    kinematic_states_counts: dict[EffectorKinematicState, int] = {
        EffectorKinematicState.CHASING: 0,
        EffectorKinematicState.TRACKING: 0,
    }
    neutralized: int
    time_step: float = 0.0
    marker: str = "D"
    color_map: dict[EffectorWeaponState, str] = {
        EffectorWeaponState.READY: "cyan",
        EffectorWeaponState.SHOOTING: "magenta",
        EffectorWeaponState.RECHARGING: "yellow",
    }
    aiming_line_length: int = 225
    aiming_line_style_map: dict[EffectorKinematicState, str] = {
        EffectorKinematicState.CHASING: "--",
        EffectorKinematicState.TRACKING: "-",
    }

    def __init__(
        self,
        identifier: str,
        location: Vector3D,
        shooting_time: float,
        max_angular_speeds: list[float],
    ):
        self.id = identifier
        self.location = location
        self.shooting_time = shooting_time
        self.max_angular_speeds = [speed * math.pi for speed in max_angular_speeds]
        self._set_rendering_style()

    def reset(self, time_step: float) -> None:
        """
        Reset the effector.

        Parameters
        ----------
        time_step: float
            The time step.
        """
        self.aiming = [0, 0]  # Azimuth - Elevation
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

    def assign_target(self, target_position: Vector3D) -> None:
        """
        Assign a target to the effector.

        Parameters
        ----------
        target_position: Vector3D
            The position of the target.
        """
        azimuth, elevation, _, distance_xy = calculate_spherical_coordinates(
            target_position, self.location
        )
        if distance_xy < 0.01:
            azimuth = self.aiming[0]

        # Limit Az-El by max angular speeds
        delta_azimuth, delta_elevation = calculate_delta_azimuth_delta_elevation(
            azimuth, elevation, self.aiming[0], self.aiming[1]
        )
        min_reachable_delta_az = min(
            self.max_angular_speeds[0] * self.time_step, abs(delta_azimuth)
        )

        self.aiming[0] += np.sign(delta_azimuth) * min_reachable_delta_az
        if self.aiming[0] > math.pi:
            self.aiming[0] -= 2 * math.pi
        elif self.aiming[0] < -math.pi:
            self.aiming[0] += 2 * math.pi

        self.aiming[1] += np.sign(delta_elevation) * min(
            abs(delta_elevation), self.max_angular_speeds[1] * self.time_step
        )

        if self.aiming[0] == azimuth and self.aiming[1] == elevation:
            self.kinematic_state = EffectorKinematicState.TRACKING
        else:
            self.kinematic_state = EffectorKinematicState.CHASING

        if self.weapon_state == EffectorWeaponState.READY:
            if self.kinematic_state == EffectorKinematicState.TRACKING:
                self.weapon_state = EffectorWeaponState.SHOOTING
        elif self.weapon_state == EffectorWeaponState.SHOOTING:
            self.weapon_state = EffectorWeaponState.RECHARGING

        if self.weapon_state != EffectorWeaponState.READY:
            self.shooting_ticks += 1

        if self.shooting_ticks == self.max_shooting_ticks:
            self.shooting_ticks = 0
            self.weapon_state = EffectorWeaponState.READY

        self._set_rendering_style()
        self.kinematic_states_counts[self.kinematic_state] += 1

    def _set_rendering_style(self) -> None:
        """Set the rendering style of the effector."""
        self.color = self.color_map[self.weapon_state]
        self.aiming_line_style = self.aiming_line_style_map[self.kinematic_state]


class Detection:
    """
    Detection class.

    Parameters
    ----------
    position: Vector3D
        The position of the detection.
    position_uncertainty: float
        The position uncertainty of the detection.
    chassis: ChassisType
        The chassis of the detection.
    explosive: ExplosiveType
        The explosive of the detection.
    """

    position: Vector3D
    position_uncertainty: float = 0.0
    velocity: Vector3D
    velocity_uncertainty: float = 0.0
    explosive: ExplosiveType
    chassis: ChassisType

    def __init__(
        self,
        position: Vector3D,
        position_uncertainty: float,
        chassis: ChassisType,
        explosive: ExplosiveType,
    ) -> None:
        self.position = position
        self.position_uncertainty = position_uncertainty
        self.chassis = chassis
        self.explosive = explosive
        self.marker_color = {
            ExplosiveType.LIGHT: "yellow",
            ExplosiveType.MEDIUM: "orange",
            ExplosiveType.STRONG: "red",
        }[self.explosive]
        self.marker = {
            ChassisType.LARGE: "p",
            ChassisType.MEDIUM: "s",
            ChassisType.SMALL: "^",
        }[self.chassis]


class Drone:
    """
    Drone class.

    Parameters
    ----------
    max_speed: float
        The max speed of the drone.
    explosive: ExplosiveType
        The explosive of the drone.
    chassis: ChassisType
        The chassis of the drone.
    """

    max_speed: float = 0.0
    explosive: ExplosiveType
    chassis: ChassisType
    marker_color: str = "white"
    marker: str = ""
    trajectory: list[Waypoint] = []
    potential_damage: float = 0.0
    detections: list[Detection] = []
    state: DroneState = DroneState.ACTIVE
    sensitive_zone_targeted: SensitiveZone | None = None

    def __init__(
        self, max_speed: float, explosive: ExplosiveType, chassis: ChassisType
    ) -> None:
        self.max_speed = max_speed
        self.explosive = explosive
        self.chassis = chassis
        self.marker_color = {
            ExplosiveType.LIGHT: "yellow",
            ExplosiveType.MEDIUM: "orange",
            ExplosiveType.STRONG: "red",
        }[self.explosive]
        self.marker = {
            ChassisType.LARGE: "p",
            ChassisType.MEDIUM: "s",
            ChassisType.SMALL: "^",
        }[self.chassis]

    def neutralize(self, tick: int) -> None:
        """
        Neutralize the drone.

        Parameters
        ----------
        tick: int
            The tick.
        """
        self.state = DroneState.NEUTRALIZED
        for idx in range(tick + 1, len(self.trajectory)):
            self.trajectory[idx].position.coords = self.trajectory[tick].position.coords
            self.marker_color = "cyan"
            self.marker = "x"
            self.detections[idx].position.coords = self.detections[tick].position.coords
            self.detections[idx].marker_color = "cyan"
            self.detections[idx].marker = "x"


class ScenarioRenderer:
    """
    ScenarioRenderer class.

    Parameters
    ----------
    domain_bb: BoundingBox
        The bounding box of the domain.
    sensitive_zones_bb: BoundingBox
        The bounding box of the sensitive zones.
    swarm_spawning_bb: BoundingBox
        The bounding box of the swarm spawning.
    swarm_intermediate_waypoints_bb: BoundingBox
        The bounding box of the swarm intermediate waypoints.
    effector_list: list[Effector]
        The list of effectors.
    sensitive_zones: list[SensitiveZone]
        The list of sensitive zones.
    plot_trajectories: bool
        Whether to plot the trajectories.
    plot_detections: bool
        Whether to plot the detections.
    grid: bool
        Whether to plot the grid.
    render_mode: str
        The mode to render the environment.
    """

    def __init__(
        self,
        domain_bb: BoundingBox,
        sensitive_zones_bb: BoundingBox,
        swarm_spawning_bb: BoundingBox,
        swarm_intermediate_waypoints_bb: BoundingBox,
        effector_list: list[Effector],
        sensitive_zones: list[SensitiveZone],
        plot_trajectories: bool = False,
        plot_detections: bool = False,
        grid: bool = False,
        render_mode: str = "human",
    ):
        self.render_mode = render_mode
        self.sensitive_zones = sensitive_zones

        self.master: tk.Tk | None
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

        self.fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(self.bg_color)
        self.fig.patch.set_facecolor(self.bg_color)

        # Set label colors
        self.ax.xaxis.label.set_color(self.base_line_color)
        self.ax.yaxis.label.set_color(self.base_line_color)
        self.ax.zaxis.label.set_color(self.base_line_color)

        # Set tick colors
        self.ax.tick_params(axis="x", colors=self.base_line_color)  # pyright: ignore[reportUnknownMemberType]
        self.ax.tick_params(axis="y", colors=self.base_line_color)  # pyright: ignore[reportUnknownMemberType]
        self.ax.tick_params(axis="z", colors=self.base_line_color)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        # Set color of the axes spines
        self.ax.xaxis.pane.set_edgecolor(self.base_line_color)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.yaxis.pane.set_edgecolor(self.base_line_color)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.zaxis.pane.set_edgecolor(self.base_line_color)

        # Set grid line colors
        gridcolor = self.base_line_color if grid else self.bg_color
        self.ax.xaxis._axinfo["grid"]["color"] = gridcolor  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.yaxis._axinfo["grid"]["color"] = gridcolor  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.zaxis._axinfo["grid"]["color"] = gridcolor  # pyright: ignore[reportUnknownMemberType, reportIndexIssue, reportPrivateUsage]

        # Set pane colors to black
        self.ax.xaxis.pane.set_facecolor(self.bg_color)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.yaxis.pane.set_facecolor(self.bg_color)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self.ax.zaxis.pane.set_facecolor(self.bg_color)

        # Draw domain BB
        self.ax.set_xlim(domain_bb.min.coords[0], domain_bb.max.coords[0])  # pyright: ignore[reportUnknownMemberType]
        self.ax.set_ylim(domain_bb.min.coords[1], domain_bb.max.coords[1])  # pyright: ignore[reportUnknownMemberType]
        self.ax.set_zlim(domain_bb.min.coords[2], domain_bb.max.coords[2])  # pyright: ignore[reportUnknownMemberType]

        # Store the scatter plot objects
        self.drones_scatter_plot: list[PathCollection] = []
        self.effectors_scatter_plot: list[PathCollection] = []
        self.aiming_lines: list[list[Line2D]] = []

        self.plot_cube(sensitive_zones_bb)
        self.plot_sensitive_zones(sensitive_zones)
        self.plot_cube(swarm_spawning_bb)
        # self.plot_cube(swarm_intermediate_waypoints_bb)
        plt.gca().set_aspect("equal")

        # Only create the Tk interface elements if in human render mode
        self.canvas: FigureCanvasTkAgg | FigureCanvasAgg | None = None
        if self.render_mode == "human" and self.master is not None:
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
            self.paned_window.add(self.plot_frame)  # pyright: ignore[reportUnknownMemberType]

            self.create_tree_views_frame(sensitive_zones)  # pyright: ignore[reportUnknownMemberType]

            # Add the tree frame to the PanedWindow
            self.paned_window.add(self.tree_frame)  # pyright: ignore[reportUnknownMemberType]

            # Configure grid resizing behavior
            self.master.grid_rowconfigure(0, weight=1)
            self.master.grid_columnconfigure(0, weight=1)
        else:
            self.canvas = FigureCanvasAgg(self.fig)

        self.update()

    def create_tree_views_frame(self, sensitive_zones: list[SensitiveZone]) -> None:
        """
        Create a frame to hold the Treeviews.

        Parameters
        ----------
        sensitive_zones: list[SensitiveZone]
            The list of sensitive zones.
        """
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
        self.style.configure(  # pyright: ignore[reportUnknownMemberType]
            "Treeview",
            background=self.bg_color,
            foreground=self.base_line_color,
            fieldbackground=self.bg_color,
        )
        self.style.map("Treeview", background=[("selected", "blue")])  # pyright: ignore[reportUnknownMemberType]

        # Stats
        # Title
        self.stats_tree_title = tk.Label(
            self.tree_frame, text="Stats", bg=self.bg_color, fg=self.base_line_color
        )
        self.stats_tree_title.grid(row=0, column=0, sticky="nsew")
        # Sub-Title 1
        self.stats_tree_subtitle_1 = tk.Label(
            self.tree_frame, text="Drones", bg=self.bg_color, fg=self.base_line_color
        )
        self.stats_tree_subtitle_1.grid(row=1, column=0, sticky="nsew")
        # Table
        self.drones_stats_tree = ttk.Treeview(
            self.tree_frame,
            columns=("Impacted", "Active", "Neutralized", "Damage [%]"),
            show="headings",
            height=1,
        )
        self.drones_stats_tree.grid(row=2, column=0, sticky="nsew")
        # Add headers for drone stats
        for col in self.drones_stats_tree["columns"]:
            self.drones_stats_tree.heading(col, text=col, anchor="center")
            self.drones_stats_tree.column(col, anchor="center")

        # Sub-Title 2
        self.stats_tree_subtitle_2 = tk.Label(
            self.tree_frame,
            text="Impacted Sensitive Zones",
            bg=self.bg_color,
            fg=self.base_line_color,
        )
        self.stats_tree_subtitle_2.grid(row=3, column=0, sticky="nsew")
        # Table
        columns: tuple[str, ...] = ("Base (Value: 1)",)
        for sensitive_zone in sensitive_zones:
            columns += (
                str(sensitive_zone.id) + " (Value: " + str(sensitive_zone.value) + ")",
            )
        self.zones_stats_tree = ttk.Treeview(
            self.tree_frame, columns=columns, show="headings", height=1
        )
        self.zones_stats_tree.grid(row=4, column=0, sticky="nsew")

        # Add headers for sensitive zones stats
        for col in self.zones_stats_tree["columns"]:
            self.zones_stats_tree.heading(col, text=col, anchor="center")
            self.zones_stats_tree.column(col, anchor="center")

        # Create a Treeview for the effectors (top half)
        # Title
        self.effectors_tree_title = tk.Label(
            self.tree_frame, text="Effectors", bg=self.bg_color, fg=self.base_line_color
        )
        self.effectors_tree_title.grid(row=5, column=0, sticky="nsew")
        # Table
        self.effectors_tree = ttk.Treeview(
            self.tree_frame,
            columns=(
                "ID",
                "Position (x,y,z)",
                "Neutralization Count",
                "Weapon State",
                "Kinematic State",
                "Tracking Steps [%]",
                "Chasing Steps [%]",
            ),
            show="headings",
            height=4,
        )
        self.effectors_tree.grid(row=6, column=0, sticky="nsew")
        # Scrollbar
        self.scrollbar_effectors = ttk.Scrollbar(self.tree_frame, orient="vertical")
        self.scrollbar_effectors.grid(row=6, column=1, sticky="ns")
        command_effectors: Any = self.effectors_tree.yview  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        self.scrollbar_effectors.config(command=command_effectors)
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
        self.drones_tree_title = tk.Label(
            self.tree_frame, text="Drones", bg=self.bg_color, fg=self.base_line_color
        )
        self.drones_tree_title.grid(row=7, column=0, sticky="nsew")
        # Table
        self.drones_tree = ttk.Treeview(
            self.tree_frame,
            columns=(
                "ID",
                "State",
                "Sensitive Target (ID - Value)",
                "Potential Damage",
                "Chassis",
                "Explosive",
                "Detected Chassis",
                "Detected Explosive",
            ),
            show="headings",
        )
        self.drones_tree.grid(row=8, column=0, sticky="nsew")
        # Scrollbar
        self.scrollbar_drones = ttk.Scrollbar(self.tree_frame, orient="vertical")
        self.scrollbar_drones.grid(row=8, column=1, sticky="ns")
        command_drones: Any = self.drones_tree.yview  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        self.scrollbar_drones.config(command=command_drones)
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

    def plot_effectors_aiming(self, effector_list: list[Effector]) -> None:
        """
        Plot the aiming of the effectors.

        Parameters
        ----------
        effector_list: list[Effector]
            The list of effectors.
        """
        for effector_scatter_plot in self.effectors_scatter_plot:
            effector_scatter_plot.remove()
        self.effectors_scatter_plot.clear()
        for aiming_line in self.aiming_lines:
            aiming_line[0].remove()
        self.aiming_lines.clear()

        for effector in effector_list:
            # Effector Marker
            effector_scatter_plot = self.ax.scatter(  # pyright: ignore[reportUnknownMemberType]
                [effector.location.coords[0]],
                [effector.location.coords[1]],
                [effector.location.coords[2]],  # pyright: ignore[reportArgumentType]
                s=self.marker_size,
                marker=effector.marker,
                color=effector.color,
            )
            self.effectors_scatter_plot.append(effector_scatter_plot)
            # Aiming Line
            x = [
                effector.location.coords[0],
                effector.location.coords[0]
                + effector.aiming_line_length
                * math.cos(effector.aiming[1])
                * math.cos(effector.aiming[0]),
            ]
            y = [
                effector.location.coords[1],
                effector.location.coords[1]
                + effector.aiming_line_length
                * math.cos(effector.aiming[1])
                * math.sin(effector.aiming[0]),
            ]
            z = [
                effector.location.coords[2],
                effector.location.coords[2]
                + effector.aiming_line_length * math.sin(effector.aiming[1]),
            ]
            aiming_line = self.ax.plot(  # pyright: ignore[reportUnknownMemberType]
                x, y, z, linestyle=effector.aiming_line_style, color=effector.color
            )
            self.aiming_lines.append(aiming_line)

    def plot_sensitive_zones(self, sensitive_zones: list[SensitiveZone]) -> None:
        """
        Plot the sensitive zones.

        Parameters
        ----------
        sensitive_zones: list[SensitiveZone]
            The list of sensitive zones.
        """
        for sensitive_zone in sensitive_zones:
            self.plot_circle(
                sensitive_zone.location.coords[0],
                sensitive_zone.location.coords[1],
                sensitive_zone.location.coords[2] + 1,
                sensitive_zone.radius,
                self.float_to_rgb(sensitive_zone.value),
            )

    def plot_drones_trajectories(self, trajectories: list[list[Waypoint]]) -> None:
        """
        Plot the trajectories of the drones.

        Parameters
        ----------
        trajectories: list[list[Waypoint]]
            The list of trajectories.
        """
        if self.plot_trajectories:
            for trajectory in trajectories:
                x = [
                    trajectory[i_waypoint].position.coords[0]
                    for i_waypoint in range(len(trajectory))
                ]
                y = [
                    trajectory[i_waypoint].position.coords[1]
                    for i_waypoint in range(len(trajectory))
                ]
                z = [
                    trajectory[i_waypoint].position.coords[2]
                    for i_waypoint in range(len(trajectory))
                ]

                self.ax.plot(x, y, z, linestyle="--", color=self.base_line_color)  # pyright: ignore[reportUnknownMemberType]

    def plot_drones_positions(self, swarm_drones_list: list[Drone], tick: int) -> None:
        """
        Plot the positions of the drones.

        Parameters
        ----------
        swarm_drones_list: list[Drone]
            The list of drones.
        tick: int
            The tick.
        """
        for drone_scatter_plot in self.drones_scatter_plot:
            drone_scatter_plot.remove()
        self.drones_scatter_plot.clear()

        # Add visualization enhancements for rgb_array mode
        if self.render_mode == "rgb_array":
            # Calculate statistics
            _, final_row, _ = self.calculate_drones_data(swarm_drones_list)
            impacted_count, active_count, neutralized_count, damage_percentage = (
                final_row
            )

            # Clear any existing text by removing all text objects
            for text in self.ax.texts:
                text.remove()

            # Add title at the top with nice background
            self.ax.text2D(  # pyright: ignore[reportUnknownMemberType]
                0.5,
                0.98,
                "Counter Drone Swarm Scenario",
                transform=self.ax.transAxes,
                fontsize=14,
                color="white",
                horizontalalignment="center",
                verticalalignment="top",
                fontweight="bold",
            )

            damage_text = f"Damage: {damage_percentage}%"
            self.ax.text2D(  # pyright: ignore[reportUnknownMemberType]
                0.05,
                0.05,
                damage_text,
                transform=self.ax.transAxes,
                fontsize=12,
                color="red",
                horizontalalignment="left",
                verticalalignment="bottom",
                bbox={"facecolor": "#000000", "alpha": 0.8},
            )

            # Impacted drones (red accent)
            self.ax.text2D(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.14,
                f"Impacted: {impacted_count}",
                transform=self.ax.transAxes,
                fontsize=10,
                color="red",
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"facecolor": "#000000", "alpha": 0.8},
            )

            # Active drones (yellow accent)
            self.ax.text2D(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.08,
                f"Active: {active_count}",
                transform=self.ax.transAxes,
                fontsize=10,
                color="red",
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"facecolor": "#000000", "alpha": 0.8},
            )

            # Neutralized drones (cyan accent)
            self.ax.text2D(  # pyright: ignore[reportUnknownMemberType]
                0.98,
                0.02,
                f"Neutralized: {neutralized_count}",
                transform=self.ax.transAxes,
                fontsize=10,
                color="red",
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"facecolor": "#000000", "alpha": 0.8},
            )

        for drone in swarm_drones_list:
            idx = min(tick, len(drone.trajectory) - 1)
            drone_scatter_plot = self.ax.scatter(  # pyright: ignore[reportUnknownMemberType]
                [drone.trajectory[idx].position.coords[0]],
                [drone.trajectory[idx].position.coords[1]],
                [drone.trajectory[idx].position.coords[2]],  # pyright: ignore[reportArgumentType]
                s=self.marker_size,
                marker=drone.marker,
                color=drone.marker_color,
            )
            self.drones_scatter_plot.append(drone_scatter_plot)
            if self.plot_detections:
                drone_scatter_plot_detection = self.ax.scatter(  # pyright: ignore[reportUnknownMemberType]
                    [drone.detections[idx].position.coords[0]],
                    [drone.detections[idx].position.coords[1]],
                    [drone.detections[idx].position.coords[2]],  # pyright: ignore[reportArgumentType]
                    s=self.marker_size,
                    marker=drone.detections[idx].marker,
                    color=drone.detections[idx].marker_color,
                )
                self.drones_scatter_plot.append(drone_scatter_plot_detection)

    def float_to_rgb(self, value: float) -> tuple[float, float, float]:
        """
        Convert a float value to an RGB color.

        Parameters
        ----------
        value: float
            The value to convert.

        Returns
        -------
        tuple[float, float, float]
            The RGB color.
        """
        value = max(1, min(value, 10))
        position = (value - 1) / 9

        # Interpolate between yellow and red
        red = 1
        green = max(0, min(1, 1 - position))
        blue = 0

        return red, green, blue

    def plot_circle(
        self,
        center_x: float,
        center_y: float,
        altitude: float,
        radius: float,
        color: tuple[float, float, float],
    ) -> None:
        """
        Plot a circle.

        Parameters
        ----------
        center_x: float
            The x coordinate of the center of the circle.
        center_y: float
            The y coordinate of the center of the circle.
        altitude: float
            The altitude of the circle.
        radius: float
            The radius of the circle.
        color: str
            The color of the circle.
        """
        # Plot circles on the ground at z=0
        circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.5)  # pyright: ignore[reportPrivateImportUsage]
        self.ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=int(altitude), zdir="z")  # pyright: ignore[reportUnknownMemberType]

    def plot_square(self, square_bb: BoundingBox) -> None:
        """
        Plot a square.

        Parameters
        ----------
        square_bb: BoundingBox
            The bounding box of the square.
        """
        # Draw a square at z=0
        square = plt.Rectangle((-5, -5), 10, 10, color=self.base_line_color, alpha=0.3)  # pyright: ignore[reportPrivateImportUsage]
        self.ax.add_patch(square)
        art3d.pathpatch_2d_to_3d(square, z=-10, zdir="z")  # pyright: ignore[reportUnknownMemberType]

    def plot_cube(self, cube_bb: BoundingBox) -> None:
        """
        Plot a cube.

        Parameters
        ----------
        cube_bb: BoundingBox
            The bounding box of the cube.
        """
        # Draw a cube
        cube = art3d.Poly3DCollection(
            [
                [
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                ],  # Bottom face
                [
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                ],  # Top face
                [
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                ],  # Side faces
                [
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                ],
                [
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.min.coords[1],
                        cube_bb.min.coords[2],
                    ),
                ],
                [
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                    (
                        cube_bb.min.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.max.coords[2],
                    ),
                    (
                        cube_bb.max.coords[0],
                        cube_bb.max.coords[1],
                        cube_bb.min.coords[2],
                    ),
                ],
            ],
            color=self.base_line_color,
            alpha=0.0,
        )
        self.ax.add_collection3d(cube)  # pyright: ignore[reportUnknownMemberType]

    def calculate_drones_data(
        self, swarm_drones_list: list[Drone]
    ) -> tuple[
        list[tuple[int, str, str, float, str, str, str, str]],
        tuple[int, int, int, float],
        dict[int, int],
    ]:
        """
        Calculate the drones data.

        Parameters
        ----------
        swarm_drones_list: list[Drone]
            The list of drones.

        Returns
        -------
        rows: list[tuple[int, str, str, float, str, str, str, str]]
            The list of rows.
        final_row: tuple[int, int, int, float]
            The final row.
        zones_stats: dict[int, int]
            The zones stats.
        """
        drones_stats: dict[DroneState, int] = {}
        zones_stats: dict[int, int] = {}
        actual_damage: float = 0.0
        potential_damage: float = 0.0
        rows: list[tuple[int, str, str, float, str, str, str, str]] = []

        for idx in range(len(self.sensitive_zones) + 1):
            zones_stats[idx] = 0

        # ("ID", "State", "Chassis", "Explosive", "Detected Chassis", "Detected Explosive")
        for state in [DroneState.IMPACTED, DroneState.ACTIVE, DroneState.NEUTRALIZED]:
            drones_stats[state] = 0
            for idx, drone in enumerate(swarm_drones_list):
                if drone.state == state:
                    drones_stats[state] += 1
                    if drone.sensitive_zone_targeted is not None:
                        sensitive_zone = (
                            "ID: "
                            + str(drone.sensitive_zone_targeted.id)
                            + " - Value: "
                            + str(drone.sensitive_zone_targeted.value)
                        )
                        if state == DroneState.IMPACTED:
                            zones_stats[drone.sensitive_zone_targeted.id + 1] += 1
                    else:
                        sensitive_zone = "ID: N/A - Value: 1"
                        if state == DroneState.IMPACTED:
                            zones_stats[0] += 1
                    if state == DroneState.IMPACTED:
                        actual_damage += drone.potential_damage
                    potential_damage += drone.potential_damage
                    row = (
                        idx,
                        drone.state.name,
                        sensitive_zone,
                        drone.potential_damage,
                        drone.chassis.name,
                        drone.explosive.name,
                        drone.detections[-1].chassis.name,
                        drone.detections[-1].explosive.name,
                    )
                    rows.append(row)

        final_row = (
            drones_stats[DroneState.IMPACTED],
            drones_stats[DroneState.ACTIVE],
            drones_stats[DroneState.NEUTRALIZED],
            round(actual_damage / potential_damage * 100, 2),
        )

        return rows, final_row, zones_stats

    def update_drones_data(self, swarm_drones_list: list[Drone]) -> None:
        """
        Update the drones data.

        Parameters
        ----------
        swarm_drones_list: list[Drone]
            The list of drones.
        """
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
        for drone_row in rows:
            self.drones_tree.insert("", "end", values=drone_row, tags=(drone_row[1],))

        self.drones_stats_tree.insert("", "end", values=final_row)

        zone_row: tuple[int, ...] = ()
        for idx in range(len(self.zones_stats_tree["columns"])):
            zone_row += (zones_stats[idx],)
        self.zones_stats_tree.insert("", "end", values=zone_row)

    def update_effectors_data(self, effectors_list: list[Effector]) -> None:
        """
        Update the effectors data.

        Parameters
        ----------
        effectors_list: list[Effector]
            The list of effectors.
        """
        # Skip updating tree views in rgb_array mode
        if self.render_mode != "human":
            return

        # Clear the Treeview
        for item in self.effectors_tree.get_children():
            self.effectors_tree.delete(item)

        # ("ID", "Position", "Weapon State", "Kinematic State", "Neutralization Count")
        for idx, effector in enumerate(effectors_list):
            position = (
                str(effector.location.coords[0])
                + ", "
                + str(effector.location.coords[1])
                + ", "
                + str(effector.location.coords[2])
            )
            total_steps = max(
                1,
                effector.kinematic_states_counts[EffectorKinematicState.TRACKING]
                + effector.kinematic_states_counts[EffectorKinematicState.CHASING],
            )
            row = (
                idx,
                position,
                effector.neutralized,
                effector.weapon_state.name,
                effector.kinematic_state.name,
                round(
                    effector.kinematic_states_counts[EffectorKinematicState.TRACKING]
                    / total_steps
                    * 100,
                    2,
                ),
                round(
                    effector.kinematic_states_counts[EffectorKinematicState.CHASING]
                    / total_steps
                    * 100,
                    2,
                ),
            )
            self.effectors_tree.insert(
                "", "end", values=row, tags=(effector.weapon_state.name,)
            )

    def update(self) -> None:
        """Update the plot."""
        # Redraw the plot
        if self.canvas is not None:
            self.canvas.draw()

        # Only update Tk if in human render mode
        if self.render_mode == "human" and self.master is not None:
            self.master.update()

    def get_rgb_array(self) -> np.ndarray:
        """
        Get the RGB array of the plot.

        Returns
        -------
        buf: np.ndarray
            The RGB array.
        """
        # Draw the figure to a canvas
        buf = np.array([])
        if self.canvas is not None:
            self.canvas.draw()

            # Get the RGB buffer from the figure - for FigureCanvasAgg
            buf = np.array(self.canvas.renderer.buffer_rgba())
            # Convert RGBA to RGB
            buf = buf[:, :, :3]

        # Return the RGB array
        return buf
