#!/usr/bin/env python3
"""
Generate top-down task visualization with 3D circles around objects.
Uses habitat-llm infrastructure to properly load scenes with objects.

Usage:
    python generate_topdown_task_viz.py +episode_indices=[64]
"""

import gzip
import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Need to setup paths before imports
sys.path.insert(0, "/home/a5l/shuqing.a5l/partnr-planner")

import magnum as mn
from PIL import Image, ImageDraw, ImageFont

from habitat_llm.agent.env import EnvironmentInterface


# Colors
GREEN = mn.Color4(0.0, 1.0, 0.0, 1.0)   # Task objects
RED = mn.Color4(1.0, 0.0, 0.0, 1.0)     # Goal receptacles


def extract_task_info(episode):
    """Extract task objects and goals from episode."""
    task_objects = []
    goal_receptacles = []

    for prop in episode.get('evaluation_propositions', []):
        args = prop.get('args', {})
        func = prop.get('function_name', '')

        if 'object_handles' in args:
            task_objects.extend(args['object_handles'])

        if func in ['is_on_top', 'is_inside'] and 'receptacle_handles' in args:
            goal_receptacles.extend(args['receptacle_handles'])

    return list(set(task_objects)), list(set(goal_receptacles))


def draw_3d_highlights(sim, task_objects, goal_receptacles):
    """Draw 3D circles around objects using DebugLineRender."""
    debug_render = sim.get_debug_line_render()
    debug_render.set_line_width(6.0)  # Thicker lines for higher resolution

    rom = sim.get_rigid_object_manager()

    # Highlight task objects in GREEN
    for handle in task_objects:
        for obj_handle in rom.get_object_handles():
            if handle in obj_handle:
                obj = rom.get_object_by_handle(obj_handle)
                if obj:
                    pos = obj.translation
                    debug_render.draw_circle(
                        translation=mn.Vector3(pos[0], pos[1] + 0.1, pos[2]),
                        radius=0.25,
                        color=GREEN,
                        num_segments=32,
                        normal=mn.Vector3(0, 1, 0)
                    )
                break

    # Highlight goal receptacles in RED
    for handle in goal_receptacles:
        for obj_handle in rom.get_object_handles():
            if handle in obj_handle:
                obj = rom.get_object_by_handle(obj_handle)
                if obj:
                    pos = obj.translation
                    debug_render.draw_circle(
                        translation=mn.Vector3(pos[0], pos[1] + 0.15, pos[2]),
                        radius=0.4,
                        color=RED,
                        num_segments=32,
                        normal=mn.Vector3(0, 1, 0)
                    )
                break


def render_topdown(env_interface, height=10.0):
    """Render top-down view using existing third_rgb sensor."""
    sim = env_interface.sim

    # Get scene bounds
    pathfinder = sim.pathfinder
    if pathfinder.is_loaded:
        bounds = pathfinder.get_bounds()
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_z = (bounds[0][2] + bounds[1][2]) / 2
    else:
        center_x, center_z = 0, 0

    # Get current observations (uses existing sensors)
    obs = env_interface.get_observations()

    # Look for third_rgb sensor
    for key in obs:
        if 'third_rgb' in key.lower():
            img = obs[key]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            return img

    # Fallback to any RGB sensor
    for key in obs:
        if 'rgb' in key.lower():
            img = obs[key]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if len(img.shape) == 3:
                return img

    return None


@hydra.main(
    version_base=None,
    config_path="habitat_llm/conf",
    config_name="baselines/decentralized_zero_shot_react_summary_with_rag.yaml",
)
def main(config: DictConfig):
    """Main function to generate visualization."""

    # Override some settings
    config.num_proc = 1
    config.evaluation.save_video = False  # We'll render manually

    # Get episode indices
    episode_indices = OmegaConf.select(config, "episode_indices", default=[64])
    if isinstance(episode_indices, str):
        episode_indices = eval(episode_indices)

    print(f"Generating visualizations for episodes: {episode_indices}")

    # Load dataset to get episode info
    dataset_path = config.habitat.dataset.data_path
    if dataset_path.endswith('.gz'):
        with gzip.open(dataset_path, 'rt') as f:
            dataset = json.load(f)
    else:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

    # Create environment
    env_interface = EnvironmentInterface(config)

    for ep_idx in episode_indices:
        print(f"\n=== Processing episode index {ep_idx} ===")

        # Get episode from dataset
        if ep_idx < len(dataset['episodes']):
            episode = dataset['episodes'][ep_idx]
            episode_id = episode.get('episode_id', str(ep_idx))
            instruction = episode.get('instruction', 'No instruction')

            print(f"Episode ID: {episode_id}")
            print(f"Instruction: {instruction}")

            # Extract task info
            task_objects, goal_receptacles = extract_task_info(episode)
            print(f"Task objects: {len(task_objects)}")
            print(f"Goal receptacles: {len(goal_receptacles)}")

            # Reset to this episode
            env_interface.reset(ep_idx)

            # Get simulator
            sim = env_interface.sim

            # Draw highlights
            draw_3d_highlights(sim, task_objects, goal_receptacles)

            # Render (uses third_rgb sensor which is already configured)
            image = render_topdown(env_interface, height=12.0)

            if image is not None:
                # Save image
                output_dir = config.paths.results_dir
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'topdown_task_viz_{episode_id}.png')

                img = Image.fromarray(image[:, :, :3])

                # Add legend
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)  # Larger font for higher resolution
                except:
                    font = ImageFont.load_default()

                draw.ellipse([10, 10, 40, 40], outline=(0, 255, 0), width=4)
                draw.text((50, 14), "Task Objects", fill=(0, 255, 0), font=font)
                draw.ellipse([10, 50, 40, 80], outline=(255, 0, 0), width=4)
                draw.text((50, 54), "Goal Receptacles", fill=(255, 0, 0), font=font)

                # Add instruction
                wrapped = '\n'.join([instruction[i:i+60] for i in range(0, len(instruction), 60)])
                draw.text((10, img.height - 50), wrapped, fill=(255, 255, 255), font=font)

                img.save(output_path)
                print(f"Saved: {output_path}")
            else:
                print("Failed to render image")

    env_interface.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
