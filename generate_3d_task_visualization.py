#!/usr/bin/env python3
"""
Generate 3D task visualization with circles rendered IN the 3D scene around objects.
Uses habitat-sim's DebugLineRender to draw 3D circles/highlights around objects,
then renders a top-down view.

This creates images like the PARTNR paper figures where circles are part of the 3D render.

Usage:
    python generate_3d_task_visualization.py --scene <scene.glb> --output task_viz.png
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add habitat paths
sys.path.insert(0, "third_party/habitat-sim")
sys.path.insert(0, "third_party/habitat-lab/habitat-lab")

import magnum as mn

try:
    import habitat_sim
    from habitat_sim.gfx import DebugLineRender
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False
    print("Error: habitat_sim required. Please install habitat-sim.")
    sys.exit(1)


class Task3DVisualizer:
    """
    Visualizer that draws 3D circles around objects in the scene
    and renders top-down views.
    """

    def __init__(self, sim: habitat_sim.Simulator, resolution: Tuple[int, int] = (2048, 2048)):
        self.sim = sim
        self.resolution = resolution
        self.debug_line_render = sim.get_debug_line_render()

    def highlight_object(
        self,
        obj_position: mn.Vector3,
        radius: float = 0.3,
        color: mn.Color4 = mn.Color4(0.0, 1.0, 0.0, 1.0),  # Green
        normal: mn.Vector3 = mn.Vector3(0.0, 1.0, 0.0),  # Horizontal circle
    ) -> None:
        """
        Draw a 3D circle around an object position in the scene.

        Args:
            obj_position: 3D position of the object center
            radius: Radius of the highlight circle
            color: Color of the circle (RGBA)
            normal: Normal vector for circle orientation (default: horizontal)
        """
        self.debug_line_render.draw_circle(
            translation=obj_position,
            radius=radius,
            color=color,
            num_segments=32,
            normal=normal,
        )

    def highlight_object_by_id(
        self,
        obj_id: int,
        color: mn.Color4 = mn.Color4(0.0, 1.0, 0.0, 1.0),
        radius_scale: float = 1.5,
    ) -> None:
        """
        Highlight an object by its ID, automatically sizing the circle.

        Args:
            obj_id: The object ID in the scene
            color: Highlight color
            radius_scale: Scale factor for the radius based on object size
        """
        rom = self.sim.get_rigid_object_manager()
        obj = rom.get_object_by_id(obj_id)

        if obj is None:
            print(f"Warning: Object {obj_id} not found")
            return

        # Get object bounding box to determine circle size
        obj_bb = obj.aabb
        obj_center = obj.translation
        obj_size = max(obj_bb.size().x, obj_bb.size().z) / 2 * radius_scale

        # Draw circle slightly above the object
        circle_pos = mn.Vector3(obj_center.x, obj_center.y + 0.05, obj_center.z)

        self.highlight_object(
            obj_position=circle_pos,
            radius=obj_size,
            color=color,
        )

    def draw_arrow_3d(
        self,
        start: mn.Vector3,
        end: mn.Vector3,
        color: mn.Color4 = mn.Color4(1.0, 0.5, 0.0, 1.0),  # Orange
    ) -> None:
        """
        Draw a 3D arrow from start to end position.
        """
        # Draw main line
        self.debug_line_render.draw_transformed_line(start, end, color)

        # Draw arrowhead
        direction = (end - start).normalized()
        arrow_length = 0.15
        arrow_angle = 0.3

        # Create two lines for arrowhead
        perp1 = mn.Vector3(-direction.z, 0, direction.x).normalized()

        head1 = end - direction * arrow_length + perp1 * arrow_angle
        head2 = end - direction * arrow_length - perp1 * arrow_angle

        self.debug_line_render.draw_transformed_line(end, head1, color)
        self.debug_line_render.draw_transformed_line(end, head2, color)

    def render_topdown_view(
        self,
        height: float = 8.0,
        look_at: Optional[mn.Vector3] = None,
    ) -> np.ndarray:
        """
        Render a top-down view of the scene with all debug drawings.

        Args:
            height: Camera height above the scene
            look_at: Point to center the camera on (default: scene center)

        Returns:
            RGB image as numpy array
        """
        # Get scene bounds
        pathfinder = self.sim.pathfinder
        if pathfinder.is_loaded:
            bounds = pathfinder.get_bounds()
            if look_at is None:
                center_x = (bounds[0][0] + bounds[1][0]) / 2
                center_z = (bounds[0][2] + bounds[1][2]) / 2
                look_at = mn.Vector3(center_x, 0, center_z)
        else:
            if look_at is None:
                look_at = mn.Vector3(0, 0, 0)

        # Position agent for top-down view
        agent = self.sim.get_agent(0)
        agent_state = agent.get_state()

        # Camera position: above the look_at point
        agent_state.position = np.array([look_at.x, height, look_at.z])

        # Rotation: look straight down (-Y direction)
        # Quaternion for 90-degree rotation around X axis
        agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            -np.pi / 2, np.array([1.0, 0.0, 0.0])
        )
        agent.set_state(agent_state)

        # Get observation (this renders the scene with debug drawings)
        obs = self.sim.get_sensor_observations()

        # Return RGB image
        if "color_sensor" in obs:
            return obs["color_sensor"][:, :, :3]
        elif "rgba" in obs:
            return obs["rgba"][:, :, :3]

        return None


def create_simulator(scene_path: str, resolution: Tuple[int, int] = (2048, 2048)):
    """Create a habitat simulator configured for visualization."""

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True

    # Agent configuration with RGB sensor
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Color sensor (for rendering)
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = list(resolution)
    color_sensor_spec.position = [0.0, 0.0, 0.0]
    color_sensor_spec.hfov = 90  # Wide field of view for top-down

    agent_cfg.sensor_specifications = [color_sensor_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    return habitat_sim.Simulator(cfg)


def get_all_objects(sim: habitat_sim.Simulator) -> Dict[str, Tuple[int, mn.Vector3]]:
    """Get all objects in the scene with their IDs and positions."""
    objects = {}
    rom = sim.get_rigid_object_manager()

    for handle in rom.get_object_handles():
        obj = rom.get_object_by_handle(handle)
        if obj is not None:
            obj_id = obj.object_id
            position = obj.translation
            objects[handle] = (obj_id, position)

    return objects


def generate_task_visualization(
    scene_path: str,
    output_path: str,
    task_objects: List[str] = None,
    goal_objects: List[str] = None,
    resolution: Tuple[int, int] = (2048, 2048),
    camera_height: float = 8.0,
):
    """
    Generate a task visualization with 3D highlights.

    Args:
        scene_path: Path to the GLB scene file
        output_path: Where to save the output image
        task_objects: List of object name patterns to highlight in green
        goal_objects: List of object name patterns to highlight in red (goals)
        resolution: Image resolution
        camera_height: Height of the top-down camera
    """
    print(f"Loading scene: {scene_path}")
    sim = create_simulator(scene_path, resolution)

    visualizer = Task3DVisualizer(sim, resolution)

    # Get all objects
    all_objects = get_all_objects(sim)
    print(f"Found {len(all_objects)} objects in scene")

    # Define colors
    GREEN = mn.Color4(0.0, 1.0, 0.0, 1.0)   # Task objects
    RED = mn.Color4(1.0, 0.0, 0.0, 1.0)     # Goal receptacles
    ORANGE = mn.Color4(1.0, 0.6, 0.0, 1.0)  # Arrows
    CYAN = mn.Color4(0.0, 1.0, 1.0, 1.0)    # Secondary highlights

    # Highlight task objects (green)
    task_positions = []
    if task_objects:
        for pattern in task_objects:
            for handle, (obj_id, pos) in all_objects.items():
                if pattern.lower() in handle.lower():
                    print(f"  Highlighting task object: {handle}")
                    visualizer.highlight_object_by_id(obj_id, color=GREEN)
                    task_positions.append(pos)

    # Highlight goal objects (red)
    goal_positions = []
    if goal_objects:
        for pattern in goal_objects:
            for handle, (obj_id, pos) in all_objects.items():
                if pattern.lower() in handle.lower():
                    print(f"  Highlighting goal: {handle}")
                    visualizer.highlight_object_by_id(obj_id, color=RED, radius_scale=2.0)
                    goal_positions.append(pos)

    # Draw arrows from task objects to goals
    if task_positions and goal_positions:
        for task_pos in task_positions:
            for goal_pos in goal_positions:
                # Elevate arrow slightly
                start = mn.Vector3(task_pos.x, task_pos.y + 0.3, task_pos.z)
                end = mn.Vector3(goal_pos.x, goal_pos.y + 0.3, goal_pos.z)
                visualizer.draw_arrow_3d(start, end, color=ORANGE)

    # If no specific objects, highlight some random ones for demo
    if not task_objects and not goal_objects:
        print("No specific objects specified, highlighting first 5 objects...")
        for i, (handle, (obj_id, pos)) in enumerate(list(all_objects.items())[:5]):
            color = GREEN if i < 3 else RED
            visualizer.highlight_object_by_id(obj_id, color=color)

    # Render top-down view
    print(f"Rendering top-down view...")
    image = visualizer.render_topdown_view(height=camera_height)

    if image is not None:
        # Save image
        from PIL import Image
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        img = Image.fromarray(image)
        img.save(output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        print("Failed to render image")

    sim.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D task visualization with object highlights'
    )
    parser.add_argument(
        '--scene', type=str, required=True,
        help='Path to the scene GLB file'
    )
    parser.add_argument(
        '--output', type=str, default='task_visualization.png',
        help='Output image path'
    )
    parser.add_argument(
        '--task-objects', type=str, nargs='+', default=[],
        help='Object name patterns to highlight as task objects (green)'
    )
    parser.add_argument(
        '--goal-objects', type=str, nargs='+', default=[],
        help='Object name patterns to highlight as goals (red)'
    )
    parser.add_argument(
        '--resolution', type=int, default=2048,
        help='Image resolution'
    )
    parser.add_argument(
        '--camera-height', type=float, default=8.0,
        help='Camera height for top-down view'
    )

    args = parser.parse_args()

    generate_task_visualization(
        scene_path=args.scene,
        output_path=args.output,
        task_objects=args.task_objects,
        goal_objects=args.goal_objects,
        resolution=(args.resolution, args.resolution),
        camera_height=args.camera_height,
    )


if __name__ == "__main__":
    main()
