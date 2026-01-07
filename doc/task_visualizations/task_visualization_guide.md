# Task Visualization Guide

This guide explains how to generate top-down 3D task visualizations with highlighted objects and goal receptacles.

## Overview

The task visualization feature generates a top-down rendered image of the scene showing:
- **Green circles**: Task objects that need to be moved
- **Red circles**: Goal receptacles (target locations)
- **Legend**: Color coding explanation
- **Task instruction**: The task description at the bottom

## Output Example

![Task Visualization Example](../../outputs/habitat_llm/2026-01-03_01-16-47-rerange+spatial_matched_subtasks.json/results/rerange+spatial_matched_subtasks.json.gz/task_visualizations/task_viz_episode_1312_0.png)

## Usage

### Method 1: Run with Video Recording SLURM Script

```bash
# Run specific episodes (e.g., episode 1312 = index 64)
export EPISODE_INDICES="[64]"
sbatch all_scripts/slurm_files/run_ours_rs_mem_r_video.slurm

# Run multiple episodes
export EPISODE_INDICES="[64,65,66,67]"
sbatch all_scripts/slurm_files/run_ours_rs_mem_r_video.slurm

# Run default top 10 successful episodes
sbatch all_scripts/slurm_files/run_ours_rs_mem_r_video.slurm
```

### Method 2: Direct Bash Script

```bash
export EPISODE_INDICES="[64]"
export NUM_PROC=1
bash all_scripts/bash_files/run_ours_rs_mem_r_video.sh
```

## Configuration

The visualization is triggered when `evaluation.save_video=True` in the config.

### Episode Index Mapping

Common episode IDs to indices (for `rerange+spatial_matched_subtasks.json.gz`):
| Episode ID | Index |
|------------|-------|
| 1312 | 64 |
| 1317 | 65 |
| 1327 | 66 |
| 1378 | 67 |
| 1425 | 69 |
| 212 | 15 |
| 214 | 17 |
| 866 | 38 |
| 1471 | 71 |
| 153 | 3 |

## Output Location

Task visualizations are saved to:
```
outputs/habitat_llm/<run_timestamp>/results/<dataset>/task_visualizations/task_viz_episode_<id>_<step>.png
```

## Implementation Details

The visualization is implemented in `habitat_llm/evaluation/evaluation_runner.py`:

- `_extract_task_objects_from_episode()`: Extracts object handles from episode propositions
- `_get_object_position_by_handle()`: Gets 3D positions of objects
- `_draw_3d_task_highlights()`: Draws 3D circles using `DebugLineRender`
- `_render_topdown_snapshot()`: Renders scene from top-down camera view
- `_save_task_visualization_image()`: Saves final image with legend

### Key Dependencies

- `magnum` (mn): For 3D vector and color operations
- `habitat_sim`: For camera positioning and rendering
- `PIL`: For image composition and legend overlay
