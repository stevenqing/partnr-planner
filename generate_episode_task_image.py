#!/usr/bin/env python3
"""
Generate task visualization image for a specific episode.
Renders a top-down view of the scene with 3D circles highlighting:
- Task objects (green circles) - objects that need to be moved
- Goal receptacles (red circles) - where objects should be placed

Usage:
    python generate_episode_task_image.py --episode-id 1312 --output task_viz.png
"""

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add paths
sys.path.insert(0, "/home/a5l/shuqing.a5l/partnr-planner")
sys.path.insert(0, "/home/a5l/shuqing.a5l/partnr-planner/third_party/habitat-sim")
sys.path.insert(0, "/home/a5l/shuqing.a5l/partnr-planner/third_party/habitat-lab/habitat-lab")

import magnum as mn

try:
    import habitat_sim
    HAS_HABITAT = True
except ImportError as e:
    print(f"Error importing habitat_sim: {e}")
    HAS_HABITAT = False
    sys.exit(1)


# Colors for highlighting
GREEN = mn.Color4(0.0, 1.0, 0.0, 1.0)   # Task objects
RED = mn.Color4(1.0, 0.0, 0.0, 1.0)     # Goal receptacles
ORANGE = mn.Color4(1.0, 0.6, 0.0, 1.0)  # Arrows
YELLOW = mn.Color4(1.0, 1.0, 0.0, 1.0)  # Highlight


def load_episode_data(dataset_path: str, episode_id: str) -> Optional[Dict]:
    """Load episode data from dataset."""
    if dataset_path.endswith('.gz'):
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)

    for ep in data.get('episodes', []):
        if str(ep.get('episode_id')) == str(episode_id):
            return ep
    return None


def extract_task_objects(episode: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract task object handles and goal receptacle handles from episode.
    Returns (task_object_handles, goal_receptacle_handles)
    """
    task_objects = set()
    goal_receptacles = set()

    for prop in episode.get('evaluation_propositions', []):
        func_name = prop.get('function_name', '')
        args = prop.get('args', {})

        # Objects that need to be moved
        if 'object_handles' in args:
            for handle in args['object_handles']:
                task_objects.add(handle)

        # Goal receptacles (where objects go)
        if func_name in ['is_on_top', 'is_inside'] and 'receptacle_handles' in args:
            for handle in args['receptacle_handles']:
                goal_receptacles.add(handle)

        # For is_next_to constraints
        if func_name == 'is_next_to':
            if 'entity_handles_a' in args:
                for handle in args['entity_handles_a']:
                    task_objects.add(handle)
            if 'entity_handles_b' in args:
                for handle in args['entity_handles_b']:
                    task_objects.add(handle)

    return list(task_objects), list(goal_receptacles)


def create_sim_config(scene_path: str, resolution: Tuple[int, int] = (1024, 1024)):
    """Create habitat-sim configuration manually."""
    # Backend configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = True

    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Sensor specification
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [resolution[1], resolution[0]]
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_spec.hfov = 90

    agent_cfg.sensor_specifications = [sensor_spec]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def get_object_position_by_handle(sim: habitat_sim.Simulator, handle: str) -> Optional[mn.Vector3]:
    """Get object position by its handle."""
    rom = sim.get_rigid_object_manager()

    # Try exact match first
    obj = rom.get_object_by_handle(handle)
    if obj is not None:
        return obj.translation

    # Try partial match
    for obj_handle in rom.get_object_handles():
        if handle in obj_handle or obj_handle in handle:
            obj = rom.get_object_by_handle(obj_handle)
            if obj is not None:
                return obj.translation

    # Also check articulated objects
    aom = sim.get_articulated_object_manager()
    for obj_handle in aom.get_object_handles():
        if handle in obj_handle or obj_handle in handle:
            obj = aom.get_object_by_handle(obj_handle)
            if obj is not None:
                return obj.translation

    return None


def draw_3d_circle(debug_render, position: mn.Vector3, radius: float,
                   color: mn.Color4, height_offset: float = 0.05):
    """Draw a 3D circle around an object."""
    circle_pos = mn.Vector3(position.x, position.y + height_offset, position.z)
    debug_render.draw_circle(
        translation=circle_pos,
        radius=radius,
        color=color,
        num_segments=32,
        normal=mn.Vector3(0.0, 1.0, 0.0)  # Horizontal circle
    )


def render_topdown_view(sim: habitat_sim.Simulator, height: float = 10.0,
                        center: Optional[mn.Vector3] = None) -> np.ndarray:
    """Render a top-down view of the scene."""
    # Get scene bounds if no center specified
    if center is None:
        pathfinder = sim.pathfinder
        if pathfinder.is_loaded:
            bounds = pathfinder.get_bounds()
            center = mn.Vector3(
                (bounds[0][0] + bounds[1][0]) / 2,
                0,
                (bounds[0][2] + bounds[1][2]) / 2
            )
        else:
            center = mn.Vector3(0, 0, 0)

    # Position agent for top-down view
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = np.array([center.x, height, center.z])

    # Look straight down
    agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        -np.pi / 2, np.array([1.0, 0.0, 0.0])
    )
    agent.set_state(agent_state)

    # Render
    obs = sim.get_sensor_observations()

    if "color_sensor" in obs:
        return obs["color_sensor"][:, :, :3]
    return None


def generate_task_visualization(
    episode_id: str,
    dataset_path: str,
    output_path: str,
    scene_base_path: str,
    resolution: Tuple[int, int] = (1024, 1024),
    camera_height: float = 12.0,
):
    """Generate task visualization for an episode."""

    print(f"Loading episode {episode_id}...")
    episode = load_episode_data(dataset_path, episode_id)

    if episode is None:
        print(f"Episode {episode_id} not found in dataset")
        return None

    scene_id = episode.get('scene_id')
    instruction = episode.get('instruction', 'No instruction')

    print(f"Scene ID: {scene_id}")
    print(f"Instruction: {instruction}")

    # Extract objects
    task_objects, goal_receptacles = extract_task_objects(episode)
    print(f"Task objects: {len(task_objects)}")
    print(f"Goal receptacles: {len(goal_receptacles)}")

    # Find scene file
    scene_path = os.path.join(scene_base_path, f"{scene_id}.scene_instance.json")
    if not os.path.exists(scene_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(scene_base_path, f"../stages/{scene_id}.glb"),
            os.path.join(scene_base_path, f"{scene_id}.glb"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                scene_path = alt
                break

    print(f"Scene path: {scene_path}")

    if not os.path.exists(scene_path):
        print(f"Scene file not found: {scene_path}")
        return None

    # Create simulator
    print("Creating simulator...")
    cfg = create_sim_config(scene_path, resolution)
    sim = habitat_sim.Simulator(cfg)

    # Get debug line render
    debug_render = sim.get_debug_line_render()
    debug_render.set_line_width(3.0)

    # Find and highlight task objects (green)
    print("\nHighlighting task objects (green):")
    for handle in task_objects:
        pos = get_object_position_by_handle(sim, handle)
        if pos is not None:
            print(f"  - {handle[:50]}... at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
            draw_3d_circle(debug_render, pos, radius=0.3, color=GREEN)
        else:
            print(f"  - {handle[:50]}... NOT FOUND")

    # Find and highlight goal receptacles (red)
    print("\nHighlighting goal receptacles (red):")
    for handle in goal_receptacles:
        pos = get_object_position_by_handle(sim, handle)
        if pos is not None:
            print(f"  - {handle[:50]}... at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
            draw_3d_circle(debug_render, pos, radius=0.5, color=RED, height_offset=0.1)
        else:
            print(f"  - {handle[:50]}... NOT FOUND")

    # Render top-down view
    print("\nRendering top-down view...")
    image = render_topdown_view(sim, height=camera_height)

    if image is not None:
        from PIL import Image, ImageDraw, ImageFont

        # Convert to PIL Image
        img = Image.fromarray(image)

        # Add instruction text at bottom
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Wrap text
        max_chars = 80
        wrapped = '\n'.join([instruction[i:i+max_chars] for i in range(0, len(instruction), max_chars)])

        # Draw text background
        text_bbox = draw.textbbox((10, resolution[1] - 60), wrapped, font=font)
        draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5],
                      fill=(0, 0, 0, 180))
        draw.text((10, resolution[1] - 60), wrapped, fill=(255, 255, 255), font=font)

        # Add legend
        draw.ellipse([10, 10, 30, 30], outline=(0, 255, 0), width=3)
        draw.text((35, 12), "Task Objects", fill=(0, 255, 0), font=font)
        draw.ellipse([10, 35, 30, 55], outline=(255, 0, 0), width=3)
        draw.text((35, 37), "Goal Receptacles", fill=(255, 0, 0), font=font)

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        img.save(output_path)
        print(f"\nSaved visualization to: {output_path}")
    else:
        print("Failed to render image")

    sim.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate task visualization image')
    parser.add_argument('--episode-id', type=str, default='1312',
                       help='Episode ID to visualize')
    parser.add_argument('--dataset', type=str,
                       default='task_classification_datasets/rerange+spatial_matched_subtasks.json.gz',
                       help='Path to dataset file')
    parser.add_argument('--scene-base', type=str,
                       default='data/versioned_data/hssd-hab/scenes-partnr-filtered',
                       help='Base path for scene files')
    parser.add_argument('--output', type=str, default='task_visualization.png',
                       help='Output image path')
    parser.add_argument('--resolution', type=int, default=1024,
                       help='Image resolution')
    parser.add_argument('--camera-height', type=float, default=12.0,
                       help='Camera height for top-down view')

    args = parser.parse_args()

    generate_task_visualization(
        episode_id=args.episode_id,
        dataset_path=args.dataset,
        output_path=args.output,
        scene_base_path=args.scene_base,
        resolution=(args.resolution, args.resolution),
        camera_height=args.camera_height,
    )


if __name__ == "__main__":
    main()
