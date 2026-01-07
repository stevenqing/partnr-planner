#!/usr/bin/env python3
"""
Generate a semantic top-down map from episode planner log data.
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np

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


def parse_world_graph(graph_str: str) -> Dict[str, List[str]]:
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

    # Extract agent positions over time
    agent_0_positions = []
    agent_1_positions = []

    # Get world graph from first step
    world_graph = {}

    for i, step in enumerate(steps):
        if 'agent_positions' in step:
            pos = step['agent_positions']
            if 'agent_0' in pos:
                agent_0_positions.append(pos['agent_0'])
            if 'agent_1' in pos:
                agent_1_positions.append(pos['agent_1'])

        # Get world graph from first step that has it
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


def generate_semantic_map(episode_data: dict, output_path: str):
    """Generate and save the semantic map visualization."""
    task = episode_data['task']
    agent_0_pos = episode_data['agent_0_positions']
    agent_1_pos = episode_data['agent_1_positions']
    world_graph = episode_data['world_graph']

    # Create figure (high resolution)
    fig, ax = plt.subplots(figsize=(24, 20))

    # Calculate bounds from agent positions
    all_x = [p[0] for p in agent_0_pos + agent_1_pos]
    all_z = [p[2] for p in agent_0_pos + agent_1_pos]

    if all_x and all_z:
        x_min, x_max = min(all_x) - 5, max(all_x) + 5
        z_min, z_max = min(all_z) - 5, max(all_z) + 5
    else:
        x_min, x_max = -20, 20
        z_min, z_max = -20, 20

    # Draw room labels with estimated positions
    # We'll distribute rooms in a grid pattern based on the space
    num_rooms = len(world_graph)
    if num_rooms > 0:
        # Create a grid layout for rooms
        cols = int(np.ceil(np.sqrt(num_rooms)))
        rows = int(np.ceil(num_rooms / cols))

        x_range = x_max - x_min
        z_range = z_max - z_min

        room_width = x_range / (cols + 1)
        room_height = z_range / (rows + 1)

        for idx, (room_name, furniture) in enumerate(world_graph.items()):
            row = idx // cols
            col = idx % cols

            rx = x_min + (col + 0.5) * room_width + room_width/2
            rz = z_min + (row + 0.5) * room_height + room_height/2

            color = get_room_color(room_name)

            # Draw room as a rounded rectangle
            room_rect = FancyBboxPatch(
                (rx - room_width/3, rz - room_height/3),
                room_width * 0.6, room_height * 0.6,
                boxstyle="round,pad=0.05,rounding_size=0.5",
                facecolor=color,
                edgecolor='black',
                alpha=0.4,
                linewidth=1.5,
                zorder=1
            )
            ax.add_patch(room_rect)

            # Add room label
            display_name = room_name.replace('_', ' ').title()
            ax.text(rx, rz, display_name, fontsize=8, ha='center', va='center',
                   fontweight='bold', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            # Show furniture count
            if furniture:
                ax.text(rx, rz - room_height/4, f"({len(furniture)} items)",
                       fontsize=6, ha='center', va='top', color='gray', zorder=10)

    # Plot Agent 0 trajectory (Robot)
    if agent_0_pos:
        x0 = [p[0] for p in agent_0_pos]
        z0 = [p[2] for p in agent_0_pos]
        ax.plot(x0, z0, 'b-', linewidth=2, alpha=0.7, label='Robot trajectory', zorder=5)
        ax.scatter([x0[0]], [z0[0]], c='green', s=150, marker='o', zorder=6, label='Robot start')
        ax.scatter([x0[-1]], [z0[-1]], c='blue', s=200, marker='^', zorder=6, label='Robot end')

    # Plot Agent 1 trajectory (Human)
    if agent_1_pos:
        x1 = [p[0] for p in agent_1_pos]
        z1 = [p[2] for p in agent_1_pos]
        ax.plot(x1, z1, 'r-', linewidth=2, alpha=0.7, label='Human trajectory', zorder=5)
        ax.scatter([x1[0]], [z1[0]], c='orange', s=150, marker='o', zorder=6, label='Human start')
        ax.scatter([x1[-1]], [z1[-1]], c='red', s=200, marker='v', zorder=6, label='Human end')

    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)

    # Wrap task text
    wrapped_task = '\n'.join([task[i:i+80] for i in range(0, len(task), 80)])
    ax.set_title(f'Episode Semantic Map\nTask: {wrapped_task}', fontsize=12, pad=20)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # Add legend
    ax.legend(loc='upper right', fontsize=9)

    # Add room color legend
    legend_patches = []
    seen_types = set()
    for room_name in world_graph.keys():
        room_type = room_name.split('_')[0] if '_' in room_name else room_name
        if room_type not in seen_types:
            seen_types.add(room_type)
            color = get_room_color(room_name)
            legend_patches.append(mpatches.Patch(color=color, alpha=0.4,
                                                  label=room_type.replace('_', ' ').title()))

    if legend_patches:
        legend2 = ax.legend(handles=legend_patches, loc='upper left',
                           fontsize=7, title='Room Types', title_fontsize=8)
        ax.add_artist(legend2)
        # Re-add the main legend
        ax.legend(loc='upper right', fontsize=9)

    # Add stats text
    stats_text = f"Total steps: {len(agent_0_pos)}\nRooms: {len(world_graph)}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save figure (high resolution)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved semantic map to: {output_path}")


def main():
    # Default path
    base_dir = "/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2026-01-02_17-22-20-rerange+spatial_matched_subtasks.json"
    json_path = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/planner-log/planner-log-episode_1312_0.json")

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    # Output path
    output_dir = os.path.join(base_dir, "results/rerange+spatial_matched_subtasks.json.gz/maps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "semantic-map-episode_1312_0.png")

    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Reading episode data from: {json_path}")
    episode_data = extract_episode_data(json_path)

    print(f"Task: {episode_data['task']}")
    print(f"Agent 0 positions: {len(episode_data['agent_0_positions'])} steps")
    print(f"Agent 1 positions: {len(episode_data['agent_1_positions'])} steps")
    print(f"Rooms found: {len(episode_data['world_graph'])}")

    generate_semantic_map(episode_data, output_path)


if __name__ == "__main__":
    main()
