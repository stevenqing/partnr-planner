#!/usr/bin/env python3
"""
Generate a realistic top-down map from habitat-sim scene.
Run with: conda activate habitat-llm && python generate_realistic_map.py
"""

import json
import os
import sys
import numpy as np

# Try to import habitat-sim
try:
    import habitat_sim
    from habitat_sim.utils.common import d3_40_colors_rgb
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False
    print("habitat_sim not available, will use fallback method")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Color palette for rooms
ROOM_COLORS = {
    'kitchen': '#FF6B6B',
    'dining_room': '#4ECDC4',
    'living_room': '#45B7D1',
    'bedroom': '#96CEB4',
    'bathroom': '#FFEAA7',
    'office': '#DDA0DD',
    'closet': '#D3D3D3',
    'hallway': '#E8E8E8',
    'garage': '#B0C4DE',
    'laundryroom': '#F0E68C',
    'entryway': '#FFB6C1',
    'tv': '#87CEEB',
    'outdoor': '#98FB98',
    'toilet': '#FFF8DC',
    'other_room': '#D8BFD8',
    'unknown': '#C0C0C0',
}


def get_room_color(room_name: str) -> str:
    """Get color for a room based on its type."""
    room_lower = room_name.lower()
    for key, color in ROOM_COLORS.items():
        if key in room_lower:
            return color
    return ROOM_COLORS['unknown']


def generate_topdown_map_habitat(scene_path: str, height: float = 0.1, resolution: int = 4096):
    """Generate top-down map using habitat-sim."""
    if not HAS_HABITAT:
        return None, None

    # Create simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)
        pathfinder = sim.pathfinder

        if not pathfinder.is_loaded:
            print("Pathfinder not loaded")
            sim.close()
            return None, None

        # Get bounds
        bounds = pathfinder.get_bounds()

        # Generate top-down map
        meters_per_pixel = max(
            (bounds[1][0] - bounds[0][0]) / resolution,
            (bounds[1][2] - bounds[0][2]) / resolution
        )

        top_down_map = pathfinder.get_topdown_view(
            meters_per_pixel=meters_per_pixel,
            height=height
        )

        sim.close()
        return top_down_map, bounds

    except Exception as e:
        print(f"Error loading scene: {e}")
        return None, None


def parse_world_graph(graph_str: str) -> dict:
    """Parse the world graph string to extract rooms and furniture."""
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


def extract_episode_data(json_path: str) -> dict:
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


def generate_realistic_map(episode_data: dict, output_path: str, topdown_map=None, bounds=None):
    """Generate realistic semantic map visualization."""
    task = episode_data['task']
    agent_0_pos = episode_data['agent_0_positions']
    agent_1_pos = episode_data['agent_1_positions']
    world_graph = episode_data['world_graph']

    # Calculate bounds from agent positions
    all_x = [p[0] for p in agent_0_pos + agent_1_pos]
    all_z = [p[2] for p in agent_0_pos + agent_1_pos]

    if bounds is not None:
        x_min, x_max = bounds[0][2], bounds[1][2]
        z_min, z_max = bounds[0][0], bounds[1][0]
    elif all_x and all_z:
        margin = 2
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        z_min, z_max = min(all_z) - margin, max(all_z) + margin
    else:
        x_min, x_max = -20, 20
        z_min, z_max = -20, 20

    # Create figure (high resolution)
    fig, ax = plt.subplots(figsize=(24, 20))

    # Draw top-down map as background if available
    if topdown_map is not None:
        # Create colored version
        colored_map = np.ones((*topdown_map.shape, 3), dtype=np.uint8) * 255
        colored_map[topdown_map == 1] = [200, 200, 200]  # Navigable (light gray)
        colored_map[topdown_map == 0] = [255, 255, 255]  # Non-navigable (white)
        colored_map[topdown_map == 2] = [100, 100, 100]  # Border (dark gray)

        ax.imshow(colored_map, extent=[x_min, x_max, z_min, z_max],
                  origin='lower', alpha=0.8, zorder=0)
    else:
        # Draw a grid background
        ax.set_facecolor('#f5f5f5')

    # Estimate room positions based on furniture clustering
    # Group furniture by room and estimate centroid
    room_centroids = {}

    # Try to estimate room positions from agent trajectory
    if agent_0_pos:
        # Divide the space into a grid and assign rooms
        x_range = x_max - x_min
        z_range = z_max - z_min

        num_rooms = len(world_graph)
        if num_rooms > 0:
            cols = int(np.ceil(np.sqrt(num_rooms)))
            rows = int(np.ceil(num_rooms / cols))

            cell_w = x_range / cols
            cell_h = z_range / rows

            for idx, (room_name, furniture) in enumerate(world_graph.items()):
                row = idx // cols
                col = idx % cols

                cx = x_min + (col + 0.5) * cell_w
                cz = z_min + (row + 0.5) * cell_h
                room_centroids[room_name] = (cx, cz, cell_w * 0.8, cell_h * 0.8)

    # Draw room regions
    for room_name, (cx, cz, w, h) in room_centroids.items():
        color = get_room_color(room_name)

        # Draw room as rounded rectangle
        rect = plt.Rectangle((cx - w/2, cz - h/2), w, h,
                            facecolor=color, edgecolor='black',
                            alpha=0.35, linewidth=1.5, zorder=1)
        ax.add_patch(rect)

        # Add room label
        display_name = room_name.replace('_', ' ').title()
        ax.text(cx, cz, display_name, fontsize=7, ha='center', va='center',
               fontweight='bold', zorder=10,
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85))

        # Show furniture count
        furniture = world_graph.get(room_name, [])
        if furniture:
            ax.text(cx, cz - h/4, f"({len(furniture)} items)",
                   fontsize=5, ha='center', va='top', color='gray', zorder=10)

    # Plot Agent 0 trajectory (Robot) - with gradient color
    if agent_0_pos:
        x0 = [p[0] for p in agent_0_pos]
        z0 = [p[2] for p in agent_0_pos]

        # Draw trajectory with color gradient (darker = more recent)
        points = np.array([x0, z0]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use simple line for now
        ax.plot(x0, z0, 'b-', linewidth=1.5, alpha=0.6, label='Robot path', zorder=4)
        ax.scatter([x0[0]], [z0[0]], c='green', s=200, marker='o',
                  edgecolors='darkgreen', linewidths=2, zorder=6, label='Robot start')
        ax.scatter([x0[-1]], [z0[-1]], c='blue', s=250, marker='^',
                  edgecolors='darkblue', linewidths=2, zorder=6, label='Robot end')

    # Plot Agent 1 trajectory (Human)
    if agent_1_pos:
        x1 = [p[0] for p in agent_1_pos]
        z1 = [p[2] for p in agent_1_pos]

        ax.plot(x1, z1, 'r-', linewidth=1.5, alpha=0.6, label='Human path', zorder=4)
        ax.scatter([x1[0]], [z1[0]], c='orange', s=200, marker='o',
                  edgecolors='darkorange', linewidths=2, zorder=6, label='Human start')
        ax.scatter([x1[-1]], [z1[-1]], c='red', s=250, marker='v',
                  edgecolors='darkred', linewidths=2, zorder=6, label='Human end')

    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)

    wrapped_task = '\n'.join([task[i:i+90] for i in range(0, len(task), 90)])
    ax.set_title(f'Episode Floor Plan\n{wrapped_task}', fontsize=11, pad=15)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)

    ax.grid(True, alpha=0.2, linestyle='-', color='gray')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Stats
    stats_text = f"Steps: {len(agent_0_pos)} | Rooms: {len(world_graph)}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved map to: {output_path}")


def main():
    base_dir = "/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2026-01-02_17-22-20-rerange+spatial_matched_subtasks.json"
    json_path = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/planner-log/planner-log-episode_1312_0.json")

    output_dir = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/maps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "realistic-map-episode_1312_0.png")

    print(f"Reading episode data...")
    episode_data = extract_episode_data(json_path)

    print(f"Task: {episode_data['task']}")
    print(f"Agent 0: {len(episode_data['agent_0_positions'])} positions")
    print(f"Agent 1: {len(episode_data['agent_1_positions'])} positions")
    print(f"Rooms: {len(episode_data['world_graph'])}")

    # Try to load actual scene map
    topdown_map = None
    bounds = None

    if HAS_HABITAT:
        # Try to find scene path
        scene_dir = "/home/a5l/shuqing.a5l/partnr-planner/data/hssd-hab/stages"
        scene_files = [f for f in os.listdir(scene_dir) if f.endswith('.glb')]
        if scene_files:
            scene_path = os.path.join(scene_dir, scene_files[0])
            print(f"Loading scene from: {scene_path}")
            topdown_map, bounds = generate_topdown_map_habitat(scene_path)

    generate_realistic_map(episode_data, output_path, topdown_map, bounds)


if __name__ == "__main__":
    main()
