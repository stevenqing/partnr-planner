#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This module contains the base EvaluationRunner class and related utilities for running evaluations of LLM-based agents in Habitat environments.
It provides functionality for initializing agents, running episodes, collecting metrics, and storing evaluation results.
The module includes classes for tracking action and state history during evaluation runs.
"""

import copy
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import attr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw, ImageFont

# Import magnum for 3D rendering
try:
    import magnum as mn
    HAS_MAGNUM = True
except ImportError:
    HAS_MAGNUM = False

from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_llm.planner.planner import Planner
from habitat_llm.utils import cprint, rollout_print
from habitat_llm.utils.sim import init_agents
from habitat_llm.world_model import Entity, WorldGraph, Room, Furniture, Receptacle

# Import map utilities from habitat-lab
try:
    from habitat.utils.visualizations.maps import (
        get_topdown_map,
        colorize_topdown_map,
        to_grid,
        draw_agent,
        TOP_DOWN_MAP_COLORS,
    )
    HAS_MAP_UTILS = True
except ImportError:
    HAS_MAP_UTILS = False

# Color palette for rooms (distinct colors for up to 12 rooms)
ROOM_COLORS = [
    (0.85, 0.35, 0.35, 0.3),  # Red
    (0.35, 0.55, 0.85, 0.3),  # Blue
    (0.35, 0.85, 0.35, 0.3),  # Green
    (0.85, 0.85, 0.35, 0.3),  # Yellow
    (0.85, 0.35, 0.85, 0.3),  # Magenta
    (0.35, 0.85, 0.85, 0.3),  # Cyan
    (0.85, 0.55, 0.35, 0.3),  # Orange
    (0.55, 0.35, 0.85, 0.3),  # Purple
    (0.55, 0.85, 0.55, 0.3),  # Light Green
    (0.85, 0.55, 0.85, 0.3),  # Pink
    (0.55, 0.85, 0.85, 0.3),  # Light Cyan
    (0.75, 0.75, 0.55, 0.3),  # Khaki
]


@attr.s(auto_attribs=True)
class ActionHistoryElement:
    """
    A class used to represent an element of action history.

    :param action: A tuple representing the action taken of format (Action Type, Action Args).
    :param timestamp: The timestamp at which the action was taken.
    :param agent_uid: The unique identifier of the agent who took the action.
    :param response: The response or feedback received after taking the action.
    :param world_graph: A dictionary mapping agent IDs to their world graph states at this point.
    :param info: Additional information dictionary containing metadata about the action.
    """

    action: tuple
    timestamp: int
    agent_uid: int
    response: str = ""
    world_graph: Dict[int, WorldGraph] = None
    info: dict = attr.ib(factory=dict)

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return f"{self.action[0]}[{self.action[1]}]"


@attr.s(auto_attribs=True)
class StateHistoryElement:
    """
    A class used to represent an element of state history.

    :param state: A string representing the state of the agent.
    :param timestamp: The timestamp of the state representation
    :param agent_uid: The unique identifier of the agent in the recorded state.
    """

    state: str
    timestamp: int
    agent_uid: int

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return self.state


# Evaluation runner, will go over episodes, run planners and store necessary data.
# Stores an episode, information about the agents and planners and uses them to run through an
# episode and store necessary data.
class EvaluationRunner:
    def __init__(
        self,
        evaluation_runner_config_arg,
        env_interface_arg: EnvironmentInterface,
        dump_world_graph: bool = False,
    ):
        """
        Initialize EvaluationRunner

        :param evaluation_runner_config_arg: The experiment configuration, including config of the agents and planners.
        :param env_interface_arg: The environment instance
        :param dump_world_graph: Whether to dump the world graph to a file.
        """
        self.env_interface = env_interface_arg
        self.evaluation_runner_config = evaluation_runner_config_arg
        self.TRUNCATE_LENGTH = self.evaluation_runner_config.truncate_length

        dataset_file = self.env_interface.conf.habitat.dataset.data_path.split("/")[-1]
        results_dir = self.env_interface.conf.paths.results_dir
        self.output_dir = f"{results_dir}/{dataset_file}/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Declare container to store agent positions
        self.agent_positions: List[Any] = []
        self.object_nodes: List[Entity] = []

        # Containers for semantic overlay visualization
        self.room_nodes: List[Entity] = []
        self.furniture_nodes: List[Entity] = []
        self._topdown_map_cache: Optional[np.ndarray] = None
        self._map_bounds: Optional[Tuple] = None

        # Declare a container for storing unique agents
        self.agents: Dict[int, Agent] = {}

        self.episode_filename = ""
        self.current_instruction = ""

        # Initialize the agents
        self.__initialize_agents()

        self.planner: Union[Dict[int, Planner], Planner] = {}
        self._initialize_planners()

        # Initialize the debug video util
        self.dvu = DebugVideoUtil(self.env_interface, self.output_dir)
        self._write_out_world_graph: bool = dump_world_graph
        self._world_graph_write_out_frequency = 5

    def _initialize_planners(self):
        """
        Initialize the planners
        """
        raise NotImplementedError

    # Method to initialize the agents based in the config
    def __initialize_agents(self) -> None:
        """
        Initialize agents based on config.
        """
        agents = init_agents(self.evaluation_runner_config.agents, self.env_interface)
        for agent in agents:
            self.agents[agent.uid] = agent
            agent._dry_run = self.env_interface._dry_run
            cprint(f"successfully added agent with UID : {agent.uid}", "green")
        print("finished initializing agents!")

    # Method to print the object
    def __str__(self) -> str:
        """
        Return string with state of the evaluator
        """
        planner_type = type(self.planner)
        out = f"Centralized Planner: {planner_type}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out

    @property
    def agent_list(self) -> str:
        """Returns a string listing the agent's uid"""
        return str([agent.uid for agent in self.agents.values()])

    @property
    def tool_list(self) -> List[str]:
        """Returns a list of unique tool names across all agents.

        :return: List of tool names as strings
        """
        tool_set = set()
        for agent in self.agents.values():
            for tool in agent.tools.values():
                tool_set.add(tool.name)

        return list(tool_set)

    def reset(self) -> None:
        """Reset metrics and stats to be ready for the next episode."""

        # Clear the frames to make sure that
        # video for next episode does no have frames from previous run
        self.dvu.frames.clear()

        # Clear containers used for top-down video generation
        self.agent_positions.clear()
        self.object_nodes.clear()
        self.room_nodes.clear()
        self.furniture_nodes.clear()
        self._topdown_map_cache = None
        self._map_bounds = None

        # Reset filenames
        self.episode_filename = ""
        self.current_instruction = ""

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

    @property
    def agent_descriptions(self) -> str:
        """Returns a string listing the descriptions of all agents

        :return: A concatenated string of all agent descriptions
        """
        out = ""
        for agent in self.agents.values():
            out += agent.agent_description
        return out

    def _get_topdown_map_image(self) -> Optional[np.ndarray]:
        """
        Get the top-down navigability map from the simulator.
        Caches the result for reuse across frames.

        :return: RGB image of the top-down map, or None if not available
        """
        if self._topdown_map_cache is not None:
            return self._topdown_map_cache

        if not HAS_MAP_UTILS:
            return None

        try:
            sim = self.env_interface.sim
            pathfinder = sim.pathfinder

            # Get agent height for the map
            agent_pos = sim.agents_mgr[0].articulated_agent.base_pos
            height = agent_pos[1]

            # Get the top-down map
            top_down_map = get_topdown_map(
                pathfinder,
                height=height,
                map_resolution=512,
                draw_border=True,
            )

            # Colorize the map
            colored_map = colorize_topdown_map(top_down_map)

            # Store bounds for coordinate conversion
            lower_bound, upper_bound = pathfinder.get_bounds()
            self._map_bounds = (lower_bound, upper_bound, top_down_map.shape)

            self._topdown_map_cache = colored_map
            return colored_map
        except Exception as e:
            print(f"Warning: Could not generate top-down map: {e}")
            return None

    def _world_to_map_coords(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map pixel coordinates.

        :param world_x: X coordinate in world space
        :param world_z: Z coordinate in world space
        :return: (map_x, map_y) pixel coordinates
        """
        if self._map_bounds is None:
            return (0, 0)

        lower_bound, upper_bound, map_shape = self._map_bounds

        # Calculate grid size
        grid_size_x = abs(upper_bound[2] - lower_bound[2]) / map_shape[0]
        grid_size_z = abs(upper_bound[0] - lower_bound[0]) / map_shape[1]

        # Convert to grid coordinates
        map_x = int((world_x - lower_bound[2]) / grid_size_x)
        map_y = int((world_z - lower_bound[0]) / grid_size_z)

        return (map_x, map_y)

    def _update_td(self, frame, ax) -> None:
        """
        Function to update the top down plot for each robot position
        and detected objects over time. Includes semantic overlay for
        rooms, furniture, and navigable areas.

        :param frame: Current animation frame number
        :param ax: Matplotlib axis object to draw on
        """
        # Clear the current plot
        ax.clear()

        # Try to draw the navigability map as background
        topdown_map = self._get_topdown_map_image()
        if topdown_map is not None and self._map_bounds is not None:
            lower_bound, upper_bound, _ = self._map_bounds
            extent = [lower_bound[2], upper_bound[2], lower_bound[0], upper_bound[0]]
            ax.imshow(topdown_map, extent=extent, origin='lower', alpha=0.7, zorder=0)

        # Draw room regions with colored overlays
        if len(self.room_nodes) > 0:
            for i, room in enumerate(self.room_nodes):
                if "translation" in room.properties:
                    rx = room.properties["translation"][0]
                    rz = room.properties["translation"][2]
                    color = ROOM_COLORS[i % len(ROOM_COLORS)]
                    # Draw room as a circle (approximate region)
                    room_circle = Circle((rx, rz), radius=3.0,
                                        facecolor=color, edgecolor=color[:3] + (0.8,),
                                        linewidth=2, zorder=1)
                    ax.add_patch(room_circle)
                    # Add room label
                    room_name = room.properties.get("category", room.name)
                    ax.text(rx, rz, room_name, fontsize=8, ha='center', va='center',
                           color='black', fontweight='bold', zorder=5,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Draw furniture as squares
        if len(self.furniture_nodes) > 0:
            for furn in self.furniture_nodes:
                if "translation" in furn.properties:
                    fx = furn.properties["translation"][0]
                    fz = furn.properties["translation"][2]
                    furn_rect = Rectangle((fx - 0.3, fz - 0.3), 0.6, 0.6,
                                         facecolor=(0.6, 0.4, 0.2, 0.5),
                                         edgecolor=(0.4, 0.2, 0.1, 0.8),
                                         linewidth=1, zorder=2)
                    ax.add_patch(furn_rect)

        # Extract x and y positions for the current frame
        x = [position[0] for position in self.agent_positions[: frame + 1]]
        y = [position[2] for position in self.agent_positions[: frame + 1]]

        # Extract object x and y
        x_obj = [obj.properties["translation"][0] for obj in self.object_nodes[frame]]
        y_obj = [obj.properties["translation"][2] for obj in self.object_nodes[frame]]
        names = [obj.name for obj in self.object_nodes[frame]]

        # Plot the robot's path with a thicker line
        ax.plot(x, y, marker=".", linestyle="-", color="blue", linewidth=2,
                markersize=4, zorder=4, label="Agent trajectory")

        # Mark start and current position
        if len(x) > 0:
            ax.scatter([x[0]], [y[0]], marker="o", color="green", s=100,
                      zorder=5, label="Start")
            ax.scatter([x[-1]], [y[-1]], marker="^", color="blue", s=150,
                      zorder=5, label="Current")

        # Plot the objects
        ax.scatter(x_obj, y_obj, marker="*", color="red", s=100, zorder=3, label="Objects")

        # Add text near each object point
        for _, txt in enumerate(zip(x_obj, y_obj, names)):
            ax.text(txt[0] + 0.2, txt[1] + 0.2, txt[2], color="darkred",
                   fontsize=7, ha="left", va="bottom", zorder=5)

        # Set labels and title
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Z (meters)")
        ax.set_title(f"Top-Down View with Semantic Overlay (Frame {frame})")

        # Set axis limits based on map bounds or default
        if self._map_bounds is not None:
            lower_bound, upper_bound, _ = self._map_bounds
            ax.set_xlim(lower_bound[2] - 1, upper_bound[2] + 1)
            ax.set_ylim(lower_bound[0] - 1, upper_bound[0] + 1)
        else:
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        # Add legend
        ax.legend(loc='upper right', fontsize=7)

    def _store_for_top_down_viz(self, agent_uid: Optional[int] = None) -> None:
        """
        Stores agent position and world graph object data for top-down visualization.
        Also captures room and furniture data for semantic overlay.

        :param agent_uid: Optional ID of agent whose perspective to use. If None, uses full observability.
        """
        world_graph = None
        if agent_uid is not None:
            world_graph = self.env_interface.world_graph[agent_uid]
        else:
            print(
                "No agent_uid provided. Code will generate top-down visualization from full-observability perspective"
            )
            world_graph = self.env_interface.full_world_graph
        sim = self.env_interface.sim
        self.agent_positions.append(
            sim.agents_mgr[agent_uid].articulated_agent.base_pos
        )

        self.object_nodes.append(world_graph.get_all_objects())

        # Capture room and furniture data (only once, on first frame)
        if len(self.room_nodes) == 0:
            try:
                self.room_nodes = world_graph.get_all_rooms()
            except Exception:
                self.room_nodes = []

        if len(self.furniture_nodes) == 0:
            try:
                self.furniture_nodes = world_graph.get_all_furnitures()
            except Exception:
                self.furniture_nodes = []

    def _log_planner_data(self, planner_infos: List[Dict[str, Any]]) -> None:
        """
        Logs planner data including prompts, traces and other info to files.

        :param planner_infos: List of dictionaries containing planner information at each step
        """
        # Print logging
        print("\nLogging planner data ...")

        # Log the latest prompts and traces
        for agent in self.agents.values():
            # -----------------------------------------------
            # Save prompts
            # Contains special tokens and few shot examples
            # -----------------------------------------------
            if "prompts" in planner_infos[-1]:
                file_path_prompts = os.path.join(
                    self.output_dir,
                    "prompts",
                    str(agent.uid),
                    f"prompt-{self.episode_filename}-{str(agent.uid)}.txt",
                )

                os.makedirs(os.path.dirname(file_path_prompts), exist_ok=True)

                with open(file_path_prompts, "w") as file:
                    file.write(planner_infos[-1]["prompts"][agent.uid])

            # -----------------------------------------------
            # Save traces
            # Skips special tokens and few shot examples
            # -----------------------------------------------
            if "traces" in planner_infos[-1]:
                file_path_traces = os.path.join(
                    self.output_dir,
                    "traces",
                    str(agent.uid),
                    f"trace-{self.episode_filename}-{str(agent.uid)}.txt",
                )

                os.makedirs(os.path.dirname(file_path_traces), exist_ok=True)

                with open(file_path_traces, "w") as file:
                    file.write(planner_infos[-1]["traces"][agent.uid])

        # Log other info from planner
        file_path_json = os.path.join(
            self.output_dir,
            "planner-log",
            f"planner-log-{self.episode_filename}.json",
        )

        # write the agents_to_actions (the plan)
        if "actions_per_agent" in planner_infos[-1]:
            actions_per_agent_path = os.path.join(
                self.output_dir,
                f"plan/{self.episode_filename}.txt",
            )
            os.makedirs(os.path.dirname(actions_per_agent_path), exist_ok=True)
            with open(actions_per_agent_path, "w") as file:
                file.write(str(planner_infos[-1]["actions_per_agent"]))

        # Make directory if it doesn't exists already
        os.makedirs(os.path.dirname(file_path_json), exist_ok=True)

        # Dictionary to store final log
        planner_log: Dict[str, Any] = {
            "task": self.current_instruction,
            "steps": [],
        }

        # Declare keys to exclude
        keys_to_exclude = ["prompts", "traces", "print", "print_no_tags"]

        # Add planner info at each step
        for i, planner_info in enumerate(planner_infos):
            step_info = {
                k: v
                for k, v in sorted(planner_info.items())
                if k not in keys_to_exclude
            }
            step_info["log_index"] = i
            planner_log["steps"].append(step_info)

        with open(file_path_json, "w+") as file:
            file.write(json.dumps(planner_log))

        print("Successfully logged planner data!")
        if self.evaluation_runner_config.log_detailed_traces:
            self._save_detailed_traces()

    def _save_detailed_traces(self) -> None:
        """
        Save detailed traces to a pickle file containing instruction, action history and state history.
        """
        for actions in self.env_interface.agent_action_history.values():
            # don't check the last action because if you hit the max sim step count no result will be logged
            for action in actions[:-1]:
                if action.response in [None, ""] and action.action[0] != "Done":
                    action_history_string = "\n".join(
                        [f"{a.action}: {a.response}" for a in actions]
                    )
                    raise ValueError(
                        f"Agent {action.agent_uid} has a null response on {action.action}: Action history:\n{action_history_string}"
                    )

        file_path_detailed_trace = os.path.join(
            self.output_dir,
            "detailed_traces",
            f"detailed_trace-{self.episode_filename}.pkl",
        )
        result = {
            "instruction": self.current_instruction,
            "action_history": self.env_interface.agent_action_history,
            "state_history": self.env_interface.agent_state_history,
        }

        os.makedirs(os.path.dirname(file_path_detailed_trace), exist_ok=True)

        with open(file_path_detailed_trace, "wb") as file:
            pickle.dump(result, file)

    def _make_td_video(self) -> None:
        """
        Creates a top-down video visualization of the episode with semantic overlay.

        :param instruction: Task instruction being executed
        """
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        td_video_name = f"{self.output_dir}/videos/video-td-{self.episode_filename}.mp4"

        # Create a larger figure for better visibility
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set the number of frames in the animation
        num_frames = len(self.agent_positions)

        # Create the animation
        animation = FuncAnimation(
            fig, self._update_td, fargs=(ax,), frames=num_frames, repeat=False
        )

        # Save the animation as a video file (e.g., .mp4)
        animation.save(td_video_name, writer="ffmpeg", fps=30)
        plt.close(fig)

        # Also save the final frame as a static image
        self._save_semantic_map_image()

    def _save_semantic_map_image(self, frame: Optional[int] = None) -> None:
        """
        Save a static image of the semantic top-down map.

        :param frame: Optional frame number. If None, uses the last frame.
        """
        if len(self.agent_positions) == 0:
            return

        os.makedirs(f"{self.output_dir}/maps", exist_ok=True)

        if frame is None:
            frame = len(self.agent_positions) - 1

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Draw the semantic map at the specified frame
        self._update_td(frame, ax)

        # Save the image
        map_image_path = f"{self.output_dir}/maps/semantic-map-{self.episode_filename}.png"
        plt.savefig(map_image_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"Saved semantic map image to: {map_image_path}")

    def _extract_task_objects_from_episode(self) -> Tuple[List[str], List[str]]:
        """
        Extract task object handles and goal receptacle handles from the current episode.

        :return: Tuple of (task_object_handles, goal_receptacle_handles)
        """
        task_objects = set()
        goal_receptacles = set()

        try:
            curr_env = self.env_interface.env.env.env._env
            episode = curr_env.current_episode

            # Get evaluation propositions from episode
            propositions = getattr(episode, 'evaluation_propositions', [])

            for prop in propositions:
                # Handle both dict and object types
                if hasattr(prop, 'function_name'):
                    func_name = prop.function_name
                    args = prop.args if hasattr(prop, 'args') else {}
                elif isinstance(prop, dict):
                    func_name = prop.get('function_name', '')
                    args = prop.get('args', {})
                else:
                    continue

                # Handle args as dict or object
                if hasattr(args, '__dict__'):
                    args = vars(args)

                # Objects that need to be moved (task objects)
                obj_handles = args.get('object_handles', []) if isinstance(args, dict) else getattr(args, 'object_handles', [])
                for handle in obj_handles:
                    task_objects.add(handle)

                # Goal receptacles (where objects should be placed)
                if func_name in ['is_on_top', 'is_inside']:
                    rec_handles = args.get('receptacle_handles', []) if isinstance(args, dict) else getattr(args, 'receptacle_handles', [])
                    for handle in rec_handles:
                        goal_receptacles.add(handle)

                # For is_next_to constraints - both entities are task objects
                if func_name == 'is_next_to':
                    handles_a = args.get('entity_handles_a', []) if isinstance(args, dict) else getattr(args, 'entity_handles_a', [])
                    handles_b = args.get('entity_handles_b', []) if isinstance(args, dict) else getattr(args, 'entity_handles_b', [])
                    for handle in handles_a:
                        task_objects.add(handle)
                    for handle in handles_b:
                        task_objects.add(handle)

        except Exception as e:
            print(f"Warning: Could not extract task objects: {e}")
            import traceback
            traceback.print_exc()

        return list(task_objects), list(goal_receptacles)

    def _get_object_position_by_handle(self, handle: str) -> Optional[np.ndarray]:
        """
        Get the 3D position of an object by its handle.

        :param handle: The object handle string
        :return: numpy array [x, y, z] or None if not found
        """
        try:
            sim = self.env_interface.sim
            rom = sim.get_rigid_object_manager()

            # Try exact match first
            for obj_handle in rom.get_object_handles():
                if handle in obj_handle or obj_handle in handle:
                    obj = rom.get_object_by_handle(obj_handle)
                    if obj is not None:
                        pos = obj.translation
                        return np.array([pos[0], pos[1], pos[2]])

            # Also check articulated objects
            aom = sim.get_articulated_object_manager()
            for obj_handle in aom.get_object_handles():
                if handle in obj_handle or obj_handle in handle:
                    obj = aom.get_object_by_handle(obj_handle)
                    if obj is not None:
                        pos = obj.translation
                        return np.array([pos[0], pos[1], pos[2]])

        except Exception as e:
            print(f"Warning: Could not get position for {handle}: {e}")

        return None

    def _draw_3d_task_highlights(self, task_objects: List[str], goal_receptacles: List[str]) -> None:
        """
        Draw 3D circles around task objects and goal receptacles using DebugLineRender.

        :param task_objects: List of task object handles to highlight in green
        :param goal_receptacles: List of goal receptacle handles to highlight in red
        """
        if not HAS_MAGNUM:
            print("Warning: magnum not available, skipping 3D highlights")
            return

        try:
            sim = self.env_interface.sim
            debug_render = sim.get_debug_line_render()
            debug_render.set_line_width(4.0)

            # Colors
            GREEN = mn.Color4(0.0, 1.0, 0.0, 1.0)   # Task objects
            RED = mn.Color4(1.0, 0.0, 0.0, 1.0)     # Goal receptacles

            # Highlight task objects in GREEN
            highlighted_task = 0
            for handle in task_objects:
                pos = self._get_object_position_by_handle(handle)
                if pos is not None:
                    circle_pos = mn.Vector3(pos[0], pos[1] + 0.1, pos[2])
                    debug_render.draw_circle(
                        translation=circle_pos,
                        radius=0.25,
                        color=GREEN,
                        num_segments=32,
                        normal=mn.Vector3(0.0, 1.0, 0.0)  # Horizontal circle
                    )
                    highlighted_task += 1

            # Highlight goal receptacles in RED
            highlighted_goal = 0
            for handle in goal_receptacles:
                pos = self._get_object_position_by_handle(handle)
                if pos is not None:
                    circle_pos = mn.Vector3(pos[0], pos[1] + 0.15, pos[2])
                    debug_render.draw_circle(
                        translation=circle_pos,
                        radius=0.4,
                        color=RED,
                        num_segments=32,
                        normal=mn.Vector3(0.0, 1.0, 0.0)
                    )
                    highlighted_goal += 1

            print(f"Drew 3D highlights: {highlighted_task} task objects (green), {highlighted_goal} goals (red)")

        except Exception as e:
            print(f"Warning: Could not draw 3D highlights: {e}")

    def _render_topdown_snapshot(self, observations: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Render a true top-down (bird's eye) 3D view of the scene.
        Positions the camera directly above the scene center looking straight down.

        :param observations: Dictionary of current observations (unused, renders fresh)
        :return: RGB image as numpy array, or None if not available
        """
        try:
            import habitat_sim

            sim = self.env_interface.sim

            # Get scene bounds to find center and appropriate height
            pathfinder = sim.pathfinder
            if pathfinder.is_loaded:
                bounds = pathfinder.get_bounds()
                center_x = (bounds[0][0] + bounds[1][0]) / 2
                center_z = (bounds[0][2] + bounds[1][2]) / 2
                # Calculate height based on scene size to see whole scene
                scene_width = max(bounds[1][0] - bounds[0][0], bounds[1][2] - bounds[0][2])
                camera_height = max(10.0, scene_width * 0.7)
            else:
                center_x, center_z = 0, 0
                camera_height = 12.0

            # Get agent and save original state
            agent = sim.get_agent(0)
            original_state = agent.get_state()

            # Position camera above scene center, looking straight down
            new_state = habitat_sim.AgentState()
            new_state.position = np.array([center_x, camera_height, center_z])
            # Rotation: look straight down (-90 degrees around X axis)
            new_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                -np.pi / 2, np.array([1.0, 0.0, 0.0])
            )
            agent.set_state(new_state)

            # Render the scene
            sim_obs = sim.get_sensor_observations()

            # Restore original agent state
            agent.set_state(original_state)

            # Find RGB observation from any sensor
            for key in sim_obs:
                if 'rgb' in key.lower() or 'color' in key.lower():
                    img = sim_obs[key]
                    arr = np.array(img)
                    if len(arr.shape) >= 3:
                        print(f"Top-down render: using sensor '{key}', shape {arr.shape}")
                        return arr

            print("Warning: No RGB sensor found for top-down render")

        except Exception as e:
            print(f"Warning: Could not render top-down snapshot: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _save_task_visualization_image(self, observations: Dict[str, Any]) -> None:
        """
        Save a top-down task visualization image with 3D highlights around
        task objects (green) and goal receptacles (red).
        """
        os.makedirs(f"{self.output_dir}/task_visualizations", exist_ok=True)

        # Extract task objects and goals from episode
        task_objects, goal_receptacles = self._extract_task_objects_from_episode()
        print(f"Task visualization: {len(task_objects)} task objects, {len(goal_receptacles)} goals")

        # Draw 3D highlights
        self._draw_3d_task_highlights(task_objects, goal_receptacles)

        # Get a fresh observation with the highlights rendered
        fresh_obs = self.env_interface.get_observations()

        # Render top-down snapshot
        image = self._render_topdown_snapshot(fresh_obs)

        if image is None:
            print("Warning: Could not render task visualization image")
            return

        # Convert tensor to numpy array properly
        if hasattr(image, 'cpu'):
            image = image.cpu().numpy()
        if hasattr(image, 'numpy'):
            image = image.numpy()

        image = np.array(image)

        # Handle various tensor shapes
        # Shape could be (1, H, W, C), (H, W, C), (1, 1, H, W, C), etc.
        while len(image.shape) > 3 and image.shape[0] == 1:
            image = image[0]
        while len(image.shape) > 3 and image.shape[1] == 1:
            image = image[:, 0]

        # Ensure we have (H, W, C) format
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            elif image.shape[0] in [3, 4]:  # (C, H, W) format
                image = np.transpose(image[:3], (1, 2, 0))

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        img = Image.fromarray(image)

        # Add legend and instruction overlay
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Draw legend background
        legend_height = 70
        draw.rectangle([0, 0, 200, legend_height], fill=(0, 0, 0, 180))

        # Draw legend items
        # Green circle for task objects
        draw.ellipse([10, 10, 30, 30], outline=(0, 255, 0), width=3)
        draw.text((35, 12), "Task Objects", fill=(0, 255, 0), font=small_font)

        # Red circle for goals
        draw.ellipse([10, 40, 30, 60], outline=(255, 0, 0), width=3)
        draw.text((35, 42), "Goal Receptacles", fill=(255, 0, 0), font=small_font)

        # Add instruction at bottom
        instruction_text = self.current_instruction
        max_chars = 80
        wrapped = '\n'.join([instruction_text[i:i+max_chars] for i in range(0, len(instruction_text), max_chars)])

        # Draw instruction background
        text_y = img.height - 60
        draw.rectangle([0, text_y - 5, img.width, img.height], fill=(0, 0, 0, 180))
        draw.text((10, text_y), wrapped, fill=(255, 255, 255), font=small_font)

        # Save the image
        output_path = f"{self.output_dir}/task_visualizations/task_viz_{self.episode_filename}.png"
        img.save(output_path)
        print(f"Saved task visualization to: {output_path}")

    def initialize_instruction_metadata(
        self, instruction: str, output_name: str
    ) -> None:
        """
        Start folders where the outputs will be stored.

        :param instruction: The natural language instruction for the task to be executed. If None, uses the instruction from the current episode.
        :param output_name: Name to use for output files. If empty string, generates name from instruction text.
        """
        if instruction is None:
            # Get the instruction from the episode
            self.current_instruction = (
                self.env_interface.env.env.env._env.current_episode.instruction
            )
        else:
            self.current_instruction = instruction
        if self.evaluation_runner_config.do_print:
            cprint("Instruction:", "yellow")
            print(self.current_instruction + "\n")
        # Make hyphenated instruction for creating a filename
        if len(output_name) == 0:
            self.episode_filename = self.current_instruction.replace(" ", "-")[
                :-1
            ].lower()
        else:
            self.episode_filename = output_name
        # check if name is too long, truncate to be system-friendly
        if len(self.episode_filename) > self.TRUNCATE_LENGTH:
            self.episode_filename = self.episode_filename[: self.TRUNCATE_LENGTH]

    def get_low_level_actions(
        self, instruction: str, observations: dict, world_graph: Dict[int, WorldGraph]
    ):
        """
        Given a set of observations, gets a vector of low level actions, an info dictionary and a boolean indicating that
        the run should end.

        :param instruction: String with the instruction to execute
        :param observations: Dictionary of habitat observations
        :param world_graph: The world graph from the agent.

        :return: tuple low_level_actions, info, should_end indicating 1) a dictionary from agent id to a low level action vector
        2) a dictionary with info about high level actions, an indicator of whether the task was ended.
        """
        raise NotImplementedError

    def reset_planners(self):
        """
        Reset the planners for this evaluator
        """
        raise NotImplementedError

    def update_agent_state_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the state history stored in env_interface based on planner info.
        This includes logging states like "standing", "walking", "picking X", "placing on Y" etc.

        :param planner_info: Dictionary containing planner state information
        """
        # # Update the agent states in environment interface
        if "agent_states" in planner_info:
            for agent_uid in planner_info["agent_states"]:
                agent_state_at_t = planner_info["agent_states"][agent_uid]
                if len(self.env_interface.agent_state_history[agent_uid]) > 0:
                    agent_state_at_t_minus_1 = self.env_interface.agent_state_history[
                        agent_uid
                    ][-1]
                    if agent_state_at_t != agent_state_at_t_minus_1.state:
                        self.env_interface.agent_state_history[agent_uid].append(
                            StateHistoryElement(
                                agent_state_at_t,
                                planner_info["sim_step_count"],
                                agent_uid=agent_uid,
                            )
                        )
                else:
                    self.env_interface.agent_state_history[agent_uid].append(
                        StateHistoryElement(
                            agent_state_at_t,
                            planner_info["sim_step_count"],
                            agent_uid=agent_uid,
                        )
                    )

    def update_agent_action_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the actions history stored in env_interface based on planner info.
        This includes logging actions like "Navigate[object_id]", "Pick[object_id]" etc.

        :param planner_info: Dictionary containing planner action information
        """
        if "replan_required" not in planner_info:
            return
        # Update the agent states in environment interface
        for agent_id, value in planner_info["replanned"].items():
            if value:
                # An action must be returned if the planner replans
                assert agent_id in planner_info["high_level_actions"]
                action_history_object = ActionHistoryElement(
                    action=planner_info["high_level_actions"][agent_id],
                    timestamp=planner_info["sim_step_count"],
                    agent_uid=agent_id,
                    world_graph=copy.deepcopy(self.env_interface.world_graph),
                    info={
                        "planner_info": planner_info,
                        "log_time": time.time(),
                    },
                )

                self.env_interface.agent_action_history[agent_id].append(
                    action_history_object
                )

        # add responses the last logged action, this means the planner will replan at the next step
        if "responses" in planner_info and any(planner_info["responses"].values()):
            for agent_id, response in planner_info["responses"].items():
                # empty string response does not mean the action is over
                # skip adding the response
                if response == "":
                    continue
                # There should have been an action logged if there is a response
                assert len(self.env_interface.agent_action_history[agent_id]) > 0
                self.env_interface.agent_action_history[agent_id][
                    -1
                ].response = response
                for ah in self.env_interface.agent_action_history[agent_id]:
                    if ah.response is None or len(ah.response) == 0:
                        raise ValueError(
                            f"Agent {agent_id} has a null response on {ah.action}"
                        )

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = ""
    ) -> Dict[str, Any]:
        """
        Runs a single instruction through the planner, taking steps until the task is done.
        Stores the information using the provided output name.

        :param instruction: Optional instruction to execute. If None, uses episode instruction.
        :param output_name: Name to use for output files. If empty, derives from instruction.

        :return: Dictionary containing execution information and metrics
        """
        # Log start time
        t_0 = time.time()

        # Counter to count iterations
        # of this loop, as sim step dont increase
        # for perception tools
        total_step_count = 1

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

        # Initialize metadata
        self.initialize_instruction_metadata(instruction, output_name)
        # Initialize sensor observations
        observations = self.env_interface.get_observations()

        # Save task visualization image at the start of episode (initial state with highlights)
        if self.evaluation_runner_config.save_video:
            try:
                self._save_task_visualization_image(observations)
            except Exception as e:
                print(f"Warning: Could not save task visualization: {e}")

        # Dictionary to store info about episode execution
        # Set default metrics incase the motor skills are never called
        # and episode ends
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
            "num_steps": 0.0,
        }

        # List to store planner logs at each step
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        low_level_actions: List[Dict[str, Any]] = []
        should_end = False

        # Get max steps for progress bar
        curr_env = self.env_interface.env.env.env._env
        max_steps = curr_env._max_episode_steps

        # Create progress bar for steps within episode
        step_pbar = tqdm(total=max_steps, desc=f"Steps", unit="step", leave=False)

        # Plan until required
        while not should_end:
            # Print the llm response
            if (
                "print" in planner_info
                and len(planner_info["print"])
                and self.evaluation_runner_config.do_print
            ):
                rollout_print(planner_info["print"])
            # Execute low level actions
            if len(low_level_actions) > 0:
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                # Refresh observations
                observations = self.env_interface.parse_observations(obs)
                if self.evaluation_runner_config.save_video:
                    # Store third person frames for generating video
                    self.dvu._store_for_video(
                        observations, planner_info["high_level_actions"]
                    )

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                self.current_instruction, observations, self.env_interface.world_graph
            )

            # We terminate the episode if this loop gets stuck
            curr_env = self.env_interface.env.env.env._env

            if total_step_count > curr_env._max_episode_steps:
                should_end = True

            measure_names = [
                "auto_eval_proposition_tracker",
                "task_constraint_validation",
                "task_percent_complete",
                "task_state_success",
                "task_evaluation_log",
                "task_explanation",
            ]
            measures_to_log = [
                "task_percent_complete",
                "task_state_success",
                "task_explanation",
            ]
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in info:
                        info[measure_name] = measures[measure_name].get_metric()

            # Add performance stats and to planner_info
            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }

            # Add step count to planner_info
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = info["num_steps"]

            # Add world description to planner_info
            # on every replanning step and at the end of planning
            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                        is_human_wg=int(agent_id) == self.env_interface.human_agent_uid
                    )
                    for agent_id in range(len(self.agents))
                }

            # Update agent state and action history
            copy_planner_info = copy.deepcopy(planner_info)
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)

            # Append planner info to history
            planner_infos.append(copy_planner_info)

            # Increment while loop step count
            total_step_count += 1

            # Update progress bar
            step_pbar.update(1)
            task_pct = info.get("task_percent_complete", 0.0)
            step_pbar.set_postfix({"task_complete": f"{task_pct:.1%}"})

            if (
                self._write_out_world_graph
                and total_step_count % self._world_graph_write_out_frequency == 0
            ):
                # dump the world-graph somewhere to compare
                for agent_id in self.env_interface.world_graph:
                    filename = f"{self.env_interface.env.env.env._env.current_episode.episode_id}_wg_agent_{agent_id}_iter_{total_step_count}.txt"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, "w") as f:
                        self.env_interface.world_graph[agent_id].display_hierarchy(
                            file_handle=f
                        )
                    print(f"WG written to:\n{filepath}")

        # Close progress bar
        step_pbar.close()

        # Print
        if (
            "print" in planner_info
            and len(planner_info["print"])
            and self.evaluation_runner_config.do_print
        ):
            rollout_print(planner_info["print"])

        # Make video
        if self.evaluation_runner_config.save_video:
            self.dvu._make_video(play=False, postfix=self.episode_filename)

        # Log planner information per step
        self._log_planner_data(planner_infos)

        # Log overall time
        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info

        return info
