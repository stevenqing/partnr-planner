#!/usr/bin/env python3
"""
Generate task visualization with top-down 3D rendered scene and object annotations.
Creates images similar to the PARTNR paper figures with colored circles highlighting
objects and arrows showing task goals.

Usage:
    python generate_task_visualization.py --episode-id 1312 --dataset <path.json.gz>
"""

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to import habitat
try:
    import habitat_sim
    from habitat_sim.utils.common import d3_40_colors_rgb
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False
    print("Warning: habitat_sim not available")


def create_orthographic_sensor_spec(height: int = 512, width: int = 512):
    """Create an orthographic (top-down) camera sensor spec."""
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "ortho_rgba"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [height, width]
    sensor_spec.position = [0.0, 10.0, 0.0]  # High above scene
    sensor_spec.orientation = [-np.pi/2, 0, 0]  # Look straight down
    return sensor_spec


def render_topdown_view(sim: habitat_sim.Simulator,
                        height: float = 10.0,
                        resolution: int = 2048) -> np.ndarray:
    """
    Render a top-down view of the scene using the simulator.

    Args:
        sim: The habitat simulator instance
        height: Height above the scene to place camera
        resolution: Image resolution

    Returns:
        RGB image as numpy array
    """
    # Get scene bounds
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        return None

    bounds = pathfinder.get_bounds()
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    center_z = (bounds[0][2] + bounds[1][2]) / 2

    # Create observation agent
    agent = sim.get_agent(0)

    # Position camera above scene center looking down
    agent_state = agent.get_state()
    agent_state.position = np.array([center_x, height, center_z])
    # Look straight down
    agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        -np.pi/2, np.array([1.0, 0.0, 0.0])
    )
    agent.set_state(agent_state)

    # Get observation
    obs = sim.get_sensor_observations()

    if "color_sensor" in obs:
        return obs["color_sensor"][:, :, :3]  # RGB only
    elif "rgba" in obs:
        return obs["rgba"][:, :, :3]

    return None


def get_object_positions(sim: habitat_sim.Simulator) -> Dict[str, Tuple[float, float, float]]:
    """Get positions of all objects in the scene."""
    object_positions = {}
    rom = sim.get_rigid_object_manager()

    for obj_id in rom.get_object_handles():
        obj = rom.get_object_by_handle(obj_id)
        if obj is not None:
            pos = obj.translation
            object_positions[obj_id] = (pos[0], pos[1], pos[2])

    return object_positions


def world_to_image_coords(world_pos: Tuple[float, float, float],
                          bounds: Tuple,
                          image_size: int) -> Tuple[int, int]:
    """Convert world coordinates to image pixel coordinates."""
    lower, upper = bounds

    # Normalize to [0, 1]
    norm_x = (world_pos[0] - lower[0]) / (upper[0] - lower[0])
    norm_z = (world_pos[2] - lower[2]) / (upper[2] - lower[2])

    # Convert to pixel coordinates
    pixel_x = int(norm_x * image_size)
    pixel_y = int((1 - norm_z) * image_size)  # Flip Y for image coords

    return (pixel_x, pixel_y)


def draw_object_highlight(image: np.ndarray,
                          center: Tuple[int, int],
                          radius: int = 30,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 3,
                          label: str = None) -> np.ndarray:
    """Draw a colored circle around an object with optional label."""
    # Draw circle
    cv2.circle(image, center, radius, color, thickness)

    # Add label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        label_pos = (center[0] - label_size[0]//2, center[1] - radius - 10)

        # Draw background rectangle for label
        cv2.rectangle(image,
                     (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                     (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                     (255, 255, 255), -1)
        cv2.putText(image, label, label_pos, font, font_scale, color, 2)

    return image


def draw_arrow(image: np.ndarray,
               start: Tuple[int, int],
               end: Tuple[int, int],
               color: Tuple[int, int, int] = (255, 165, 0),
               thickness: int = 2) -> np.ndarray:
    """Draw an arrow from start to end position."""
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=0.3)
    return image


def generate_task_visualization(
    scene_path: str,
    task_objects: List[str],
    goal_receptacle: str,
    output_path: str,
    resolution: int = 2048
) -> str:
    """
    Generate a task visualization image.

    Args:
        scene_path: Path to the scene GLB file
        task_objects: List of object names involved in the task
        goal_receptacle: Name of the goal receptacle
        output_path: Where to save the output image
        resolution: Image resolution

    Returns:
        Path to the saved image
    """
    if not HAS_HABITAT:
        print("Cannot generate visualization without habitat_sim")
        return None

    # Configure simulator
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    # Configure agent with RGB sensor
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Color sensor
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [resolution, resolution]
    color_sensor_spec.position = [0.0, 0.0, 0.0]
    agent_cfg.sensor_specifications = [color_sensor_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)

        # Render top-down view
        image = render_topdown_view(sim, height=12.0, resolution=resolution)

        if image is None:
            print("Failed to render top-down view")
            sim.close()
            return None

        # Get object positions
        object_positions = get_object_positions(sim)
        bounds = sim.pathfinder.get_bounds()

        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Highlight task objects in green
        for obj_name in task_objects:
            for obj_handle, pos in object_positions.items():
                if obj_name.lower() in obj_handle.lower():
                    pixel_pos = world_to_image_coords(pos, bounds, resolution)
                    image_bgr = draw_object_highlight(
                        image_bgr, pixel_pos,
                        radius=25,
                        color=(0, 255, 0),  # Green
                        label=obj_name
                    )

        # Highlight goal receptacle in red
        for obj_handle, pos in object_positions.items():
            if goal_receptacle.lower() in obj_handle.lower():
                pixel_pos = world_to_image_coords(pos, bounds, resolution)
                image_bgr = draw_object_highlight(
                    image_bgr, pixel_pos,
                    radius=35,
                    color=(0, 0, 255),  # Red
                    thickness=4,
                    label=goal_receptacle
                )

        # Save image
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, image_bgr)
        print(f"Saved visualization to: {output_path}")

        sim.close()
        return output_path

    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_from_episode(dataset_path: str,
                          episode_id: str,
                          output_dir: str) -> str:
    """
    Generate visualization from a dataset episode.

    Args:
        dataset_path: Path to the dataset JSON file
        episode_id: Episode ID to visualize
        output_dir: Output directory for images
    """
    # Load dataset
    if dataset_path.endswith('.gz'):
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)

    # Find episode
    episode = None
    for ep in data.get('episodes', []):
        if str(ep.get('episode_id')) == str(episode_id):
            episode = ep
            break

    if episode is None:
        print(f"Episode {episode_id} not found")
        return None

    scene_id = episode.get('scene_id', '')
    instruction = episode.get('instruction', '')

    print(f"Episode {episode_id}")
    print(f"Scene: {scene_id}")
    print(f"Instruction: {instruction}")

    # Extract task objects from instruction (simple parsing)
    # This would need to be customized based on your task format

    output_path = os.path.join(output_dir, f"task_viz_episode_{episode_id}.png")

    # For now, create a placeholder visualization using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(20, 16))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5dc')  # Beige floor

    # Draw some furniture rectangles
    furniture = [
        (10, 10, 20, 15, 'Sofa', '#8B4513'),
        (50, 60, 25, 15, 'Table', '#DEB887'),
        (80, 30, 15, 20, 'Cabinet', '#A0522D'),
    ]

    for x, y, w, h, name, color in furniture:
        rect = mpatches.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=8, color='white')

    # Draw task objects (green circles)
    objects = [(25, 70, 'Vase'), (35, 72, 'Candle')]
    for x, y, name in objects:
        circle = mpatches.Circle((x, y), 5, facecolor='none', edgecolor='green', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y + 8, name, ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

    # Draw goal receptacle (red circle)
    goal_x, goal_y = 62, 67
    circle = mpatches.Circle((goal_x, goal_y), 8, facecolor='none', edgecolor='red', linewidth=3)
    ax.add_patch(circle)
    ax.text(goal_x, goal_y + 12, 'Goal: Table', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    # Draw arrow from object to goal
    ax.annotate('', xy=(goal_x - 10, goal_y), xytext=(35, 72),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    # Title with instruction
    wrapped_instruction = '\n'.join([instruction[i:i+80] for i in range(0, len(instruction), 80)])
    ax.set_title(f"Episode {episode_id}\n{wrapped_instruction}", fontsize=10, pad=10)

    ax.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved visualization to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate task visualization')
    parser.add_argument('--dataset', type=str, help='Path to dataset JSON file')
    parser.add_argument('--episode-id', type=str, help='Episode ID to visualize')
    parser.add_argument('--scene', type=str, help='Path to scene GLB file (if not using dataset)')
    parser.add_argument('--output-dir', type=str, default='task_visualizations', help='Output directory')

    args = parser.parse_args()

    if args.dataset and args.episode_id:
        generate_from_episode(args.dataset, args.episode_id, args.output_dir)
    elif args.scene:
        # Direct scene visualization
        output_path = os.path.join(args.output_dir, 'task_viz.png')
        generate_task_visualization(
            args.scene,
            task_objects=['object1', 'object2'],
            goal_receptacle='table',
            output_path=output_path
        )
    else:
        print("Please provide either --dataset with --episode-id, or --scene")
        parser.print_help()


if __name__ == "__main__":
    main()
