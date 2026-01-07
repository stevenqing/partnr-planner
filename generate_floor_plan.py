#!/usr/bin/env python3
"""
Generate a realistic floor plan from habitat-sim scene using the actual navmesh.
Run with: conda activate habitat-llm && python generate_floor_plan.py
"""

import json
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Import habitat-sim
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

ROOM_COLORS = {
    'kitchen': '#FF6B6B',
    'dining': '#4ECDC4',
    'living': '#45B7D1',
    'bedroom': '#96CEB4',
    'bathroom': '#FFEAA7',
    'office': '#DDA0DD',
    'closet': '#D3D3D3',
    'hallway': '#E8E8E8',
    'garage': '#B0C4DE',
    'laundry': '#F0E68C',
    'entry': '#FFB6C1',
    'tv': '#87CEEB',
    'outdoor': '#98FB98',
    'toilet': '#FFF8DC',
    'other': '#D8BFD8',
}


def load_room_annotations(scene_id):
    """Load room annotations from semantic config file."""
    semantic_config_path = f"/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/semantics/scenes/{scene_id}.semantic_config.json"

    if not os.path.exists(semantic_config_path):
        print(f"Semantic config not found: {semantic_config_path}")
        return []

    with open(semantic_config_path, 'r') as f:
        data = json.load(f)

    rooms = []
    for region in data.get('region_annotations', []):
        name = region.get('name', 'unknown')
        label = region.get('label', name)
        poly_loop = region.get('poly_loop', [])
        min_bounds = region.get('min_bounds', [0, 0, 0])
        max_bounds = region.get('max_bounds', [0, 0, 0])

        # Calculate centroid from polygon or bounds
        if poly_loop:
            # Use polygon centroid (x and z coordinates)
            xs = [p[0] for p in poly_loop]
            zs = [p[2] for p in poly_loop]
            centroid_x = sum(xs) / len(xs)
            centroid_z = sum(zs) / len(zs)

            # Create polygon points for drawing
            polygon_points = [(p[0], p[2]) for p in poly_loop]
        else:
            # Use bounds center
            centroid_x = (min_bounds[0] + max_bounds[0]) / 2
            centroid_z = (min_bounds[2] + max_bounds[2]) / 2
            polygon_points = None

        rooms.append({
            'name': name,
            'label': label,
            'centroid': (centroid_x, centroid_z),
            'polygon': polygon_points,
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
        })

    return rooms


def get_room_color(room_name):
    room_lower = room_name.lower()
    for key, color in ROOM_COLORS.items():
        if key in room_lower:
            return color
    return '#C0C0C0'


def create_simulator(scene_id):
    """Create habitat simulator with the specified scene."""
    scene_glb = f"/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/stages/{scene_id}.glb"
    scene_config = f"/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/stages/{scene_id}.stage_config.json"

    # Check if scene exists
    if not os.path.exists(scene_glb):
        print(f"Scene file not found: {scene_glb}")
        return None

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb
    backend_cfg.scene_dataset_config_file = "/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"

    # Enable physics for navmesh
    backend_cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)
        return sim
    except Exception as e:
        print(f"Error creating simulator: {e}")
        return None


def get_topdown_map(sim, height=0.1, resolution=4096):
    """Generate top-down navigability map."""
    pathfinder = sim.pathfinder

    if not pathfinder.is_loaded:
        # Try to recompute navmesh using the correct API
        try:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_radius = 0.2
            navmesh_settings.agent_height = 1.5

            # Use recompute_navmesh on the simulator, not pathfinder
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        except Exception as e:
            print(f"Could not recompute navmesh: {e}")
            # Try to load a pre-existing navmesh
            navmesh_path = "/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/stages/102816009.navmesh"
            if os.path.exists(navmesh_path):
                pathfinder.load_nav_mesh(navmesh_path)

    if not pathfinder.is_loaded:
        print("Pathfinder still not loaded - generating approximate map from scene bounds")
        # Fall back to using scene bounds
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        bounds = (
            np.array([scene_bb.min[0], scene_bb.min[1], scene_bb.min[2]]),
            np.array([scene_bb.max[0], scene_bb.max[1], scene_bb.max[2]])
        )
        return None, bounds

    bounds = pathfinder.get_bounds()
    print(f"Scene bounds: {bounds}")

    # Calculate meters per pixel
    x_range = bounds[1][0] - bounds[0][0]
    z_range = bounds[1][2] - bounds[0][2]
    meters_per_pixel = max(x_range, z_range) / resolution

    # Get top-down view
    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel,
        height=height
    )

    return top_down_map, bounds


def parse_world_graph(graph_str):
    """Parse the world graph string."""
    rooms = {}
    current_section = None

    for line in graph_str.split('\n'):
        line = line.strip()
        if line.startswith('Furniture:'):
            current_section = 'furniture'
            continue
        elif line.startswith('Objects:'):
            current_section = 'objects'
            continue

        if current_section == 'furniture' and ':' in line:
            parts = line.split(':', 1)
            room_name = parts[0].strip()
            furniture_str = parts[1].strip() if len(parts) > 1 else ''
            furniture_list = [f.strip() for f in furniture_str.split(',') if f.strip()]
            rooms[room_name] = furniture_list

    return rooms


def extract_episode_data(json_path):
    """Extract agent positions and world graph from planner log."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    task = data.get('task', 'Unknown task')
    steps = data.get('steps', [])

    agent_0_positions = []
    agent_1_positions = []
    world_graph = {}

    for step in steps:
        if 'agent_positions' in step:
            pos = step['agent_positions']
            if 'agent_0' in pos:
                agent_0_positions.append(pos['agent_0'])
            if 'agent_1' in pos:
                agent_1_positions.append(pos['agent_1'])

        if not world_graph and 'curr_graph' in step:
            for agent_id, graph_str in step['curr_graph'].items():
                world_graph = parse_world_graph(graph_str)
                break

    return {
        'task': task,
        'agent_0_positions': agent_0_positions,
        'agent_1_positions': agent_1_positions,
        'world_graph': world_graph,
    }


def generate_floor_plan(episode_data, output_path, topdown_map, bounds, room_annotations=None):
    """Generate floor plan visualization with actual scene layout and room labels."""
    task = episode_data['task']
    agent_0_pos = episode_data['agent_0_positions']
    agent_1_pos = episode_data['agent_1_positions']

    # Scene bounds
    x_min, x_max = bounds[0][0], bounds[1][0]
    z_min, z_max = bounds[0][2], bounds[1][2]

    # Create figure (high resolution)
    fig, ax = plt.subplots(figsize=(24, 20))

    # Colorize the top-down map
    if topdown_map is not None:
        h, w = topdown_map.shape
        colored_map = np.ones((h, w, 3), dtype=np.float32)

        # Navigable areas - light color
        colored_map[topdown_map == 1] = [0.95, 0.95, 0.92]  # Light beige for floor
        # Non-navigable - white (outside)
        colored_map[topdown_map == 0] = [1.0, 1.0, 1.0]
        # Walls/borders - dark
        colored_map[topdown_map == 2] = [0.3, 0.25, 0.2]  # Dark brown for walls

        # Add some texture to the floor
        noise = np.random.rand(h, w) * 0.015
        for i in range(3):
            colored_map[:,:,i] = np.clip(colored_map[:,:,i] + noise * (topdown_map == 1), 0, 1)

        # Draw the map
        ax.imshow(colored_map, extent=[x_min, x_max, z_min, z_max],
                  origin='lower', zorder=0)

    # Draw room polygons and labels
    if room_annotations:
        from matplotlib.patches import Polygon as MplPolygon
        import matplotlib.colors as mcolors

        for room in room_annotations:
            name = room['name']
            label = room['label']
            centroid = room['centroid']
            polygon = room['polygon']

            # Get room color
            color = get_room_color(label)
            rgb = mcolors.to_rgb(color)

            # Draw room polygon with semi-transparent fill
            if polygon and len(polygon) >= 3:
                poly_patch = MplPolygon(polygon, closed=True,
                                       facecolor=(*rgb, 0.25),
                                       edgecolor=(*rgb[:3], 0.6),
                                       linewidth=1.5, zorder=1)
                ax.add_patch(poly_patch)

            # Add room label at centroid
            display_name = label.replace('_', ' ').title()

            # Add background box for better readability
            ax.text(centroid[0], centroid[1], display_name,
                   fontsize=8, fontweight='bold',
                   ha='center', va='center',
                   color='#333333', zorder=8,
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor=color,
                            alpha=0.85,
                            linewidth=1.5))

    # Plot Agent 0 trajectory (Robot)
    if agent_0_pos:
        x0 = [p[0] for p in agent_0_pos]
        z0 = [p[2] for p in agent_0_pos]

        # Subsample for cleaner visualization
        step = max(1, len(x0) // 500)
        x0_sub = x0[::step]
        z0_sub = z0[::step]

        ax.plot(x0_sub, z0_sub, 'b-', linewidth=2, alpha=0.7, label='Robot path', zorder=4)
        ax.scatter([x0[0]], [z0[0]], c='lime', s=300, marker='o',
                  edgecolors='darkgreen', linewidths=3, zorder=6, label='Robot start')
        ax.scatter([x0[-1]], [z0[-1]], c='blue', s=350, marker='^',
                  edgecolors='darkblue', linewidths=3, zorder=6, label='Robot end')

    # Plot Agent 1 trajectory (Human)
    if agent_1_pos:
        x1 = [p[0] for p in agent_1_pos]
        z1 = [p[2] for p in agent_1_pos]

        step = max(1, len(x1) // 500)
        x1_sub = x1[::step]
        z1_sub = z1[::step]

        ax.plot(x1_sub, z1_sub, 'r-', linewidth=2, alpha=0.7, label='Human path', zorder=4)
        ax.scatter([x1[0]], [z1[0]], c='orange', s=300, marker='o',
                  edgecolors='darkorange', linewidths=3, zorder=6, label='Human start')
        ax.scatter([x1[-1]], [z1[-1]], c='red', s=350, marker='v',
                  edgecolors='darkred', linewidths=3, zorder=6, label='Human end')

    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Z (meters)', fontsize=14)

    wrapped_task = '\n'.join([task[i:i+100] for i in range(0, len(task), 100)])
    ax.set_title(f'Floor Plan - Scene 102816009\n{wrapped_task}', fontsize=13, pad=15)

    # Axis limits with margin
    margin = 1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(z_min - margin, z_max + margin)

    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    # Stats
    stats = f"Steps: {len(agent_0_pos)} | Scene: 102816009"
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Scale bar
    scale_length = 5  # meters
    scale_x = x_min + 2
    scale_y = z_min + 1
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', linewidth=3)
    ax.text(scale_x + scale_length/2, scale_y + 0.5, f'{scale_length}m',
            ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved floor plan to: {output_path}")


def main():
    scene_id = "102816009"

    base_dir = "/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2026-01-02_17-22-20-rerange+spatial_matched_subtasks.json"
    json_path = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/planner-log/planner-log-episode_1312_0.json")

    output_dir = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/maps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "floor-plan-episode_1312_0.png")

    print(f"Loading scene {scene_id}...")
    sim = create_simulator(scene_id)

    if sim is None:
        print("Failed to create simulator")
        return

    print("Generating top-down map...")
    topdown_map, bounds = get_topdown_map(sim, height=0.1, resolution=2048)

    if topdown_map is None:
        print("Could not generate navmesh-based map, will use trajectory-only visualization")
    else:
        print(f"Map shape: {topdown_map.shape}")
        print(f"Navigable cells: {np.sum(topdown_map == 1)}")
        print(f"Wall cells: {np.sum(topdown_map == 2)}")

    print("Reading episode data...")
    episode_data = extract_episode_data(json_path)
    print(f"Task: {episode_data['task']}")
    print(f"Robot positions: {len(episode_data['agent_0_positions'])}")
    print(f"Human positions: {len(episode_data['agent_1_positions'])}")

    print("Loading room annotations...")
    room_annotations = load_room_annotations(scene_id)
    print(f"Found {len(room_annotations)} rooms")

    print("Generating floor plan with room labels...")
    generate_floor_plan(episode_data, output_path, topdown_map, bounds, room_annotations)

    sim.close()
    print("Done!")


if __name__ == "__main__":
    main()
