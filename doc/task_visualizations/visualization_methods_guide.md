# Task Visualization Methods Guide
## PartnR-Planner Repository

This document summarizes all available visualization methods for tasks in this repository.

---

## 1. Video Recording (Simulation Playback)

**Best for:** Watching agent behavior, debugging trajectories, creating demos

### Description
Records third-person RGB video during simulation execution, showing both agents (human & robot) performing the task.

### How to Use
```bash
# Enable video recording in evaluation
python -m habitat_llm.examples.planner_demo \
    evaluation.save_video=True \
    episode_indices=[64,65,66] \
    num_proc=1

# Or use the prepared script
sbatch all_scripts/slurm_files/run_ours_rs_mem_r_video.slurm
```

### Output
- Location: `outputs/habitat_llm/<run>/results/<dataset>/videos/`
- Format: MP4 files with agent action overlays
- Shows: Split-screen view of both agents with action labels

### Requirements
- GPU with EGL support
- Habitat-sim environment
- ~10-30 min per episode

### Status: ✅ Available (Job 167990 submitted)

---

## 2. PrediViz (Static Scene Diagrams)

**Best for:** Paper figures, task structure visualization, constraint visualization

### Description
Generates static diagrams showing:
- Room layouts with objects and receptacles
- Object placement before/after states
- Spatial constraints (next_to, on_top)
- Temporal constraints (task ordering)

### How to Use
```bash
cd scripts/prediviz
python viz.py \
    --dataset <path_to_dataset.json.gz> \
    --metadata-dir <path_to_metadata> \
    --save-path <output_dir> \
    --episode-id <episode_id>
```

### Output
- Format: PNG/PDF figures
- Shows: Schematic room layouts, object icons, constraint arrows

### Requirements
- Episode metadata JSON files (object_to_room, recep_to_room mappings)
- Receptacle icons in `scripts/prediviz/assets/`
- Fonts in `scripts/prediviz/assets/fonts/`

### Status: ⚠️ Requires metadata files (not available for custom datasets)

---

## 3. Top-Down Map Visualization

**Best for:** Agent trajectory analysis, navigation debugging

### Description
Animated top-down view showing:
- Agent movement paths over time
- Object positions
- Room boundaries

### How to Use
```python
# In evaluation_runner.py
self._store_for_top_down_viz(agent_uid)
self._make_td_video()
```

### Output
- Location: `outputs/<run>/videos/video-td-<episode>.mp4`
- Format: MP4 animation
- Shows: Bird's eye view with agent paths and object markers

### Requirements
- Enabled during evaluation run
- Agent position tracking

### Status: ✅ Available (built into evaluation runner)

---

## 4. Episode Trace Visualization

**Best for:** Debugging LLM reasoning, analyzing action sequences

### Description
Text-based traces showing:
- LLM thoughts and reasoning
- Actions taken by each agent
- Success/failure of each action
- World state changes

### How to Use
```bash
# Traces are automatically saved during evaluation
cat outputs/habitat_llm/<run>/results/<dataset>/traces/<agent_id>/trace-*.txt
```

### Output Example
```
Task: Move the vase and candle from the shelves...
Thought: I need to explore the living room first...
Explore[living_room_1]
Result: Successful execution!
Objects: vase_0: shelves_7 in living_room_1
...
```

### Status: ✅ Available (already generated for all runs)

---

## 5. Dataset Analysis Plots

**Best for:** Dataset statistics, task distribution analysis

### Description
Statistical visualizations including:
- Task type distributions (pie charts)
- Object/receptacle frequency (bar charts)
- UpSet plots for task combinations
- Ambiguity metrics

### How to Use
```bash
cd dataset_generation/benchmark_generation/analysis
python run_dataset_analysis.py --dataset <path>
python task_type_upset_plot.py --dataset <path>
```

### Output
- Format: PDF/PNG charts
- Shows: Dataset composition and statistics

### Status: ✅ Available

---

## 6. HITL Episode Visualization

**Best for:** Human-in-the-loop data analysis

### Description
Animated playback of recorded HITL sessions with:
- Object tracking
- Agent actions
- Task completion metrics

### How to Use
```bash
cd scripts/hitl_analysis
python visualize_episode.py \
    --episodes-path <hitl_data_dir> \
    --dataset-file <dataset.json.gz>
```

### Requirements
- HITL session recordings
- Episode dataset file

### Status: ⚠️ Requires HITL data format

---

## Summary Table

| Method | Type | Best For | Status |
|--------|------|----------|--------|
| Video Recording | Video | Demos, debugging | ✅ Ready |
| PrediViz | Static | Paper figures | ⚠️ Needs metadata |
| Top-Down Map | Video | Trajectory analysis | ✅ Ready |
| Episode Traces | Text | LLM debugging | ✅ Generated |
| Dataset Analysis | Charts | Statistics | ✅ Ready |
| HITL Visualization | Video | HITL analysis | ⚠️ Needs HITL data |

---

## Quick Start: Visualize Top 10 Tasks

### Option 1: Video Recording (Recommended)
```bash
# Already submitted as job 167990
squeue -j 167990  # Check status

# Videos will be at:
# outputs/habitat_llm/<timestamp>/results/.../videos/
```

### Option 2: View Existing Traces
```bash
# View trace for episode 1312
cat outputs/habitat_llm/2025-12-31_16-18-25-rerange+spatial_matched_subtasks.json/\
    results/rerange+spatial_matched_subtasks.json.gz/traces/0/trace-episode_1312*.txt
```

### Option 3: Generate Dataset Statistics
```bash
cd dataset_generation/benchmark_generation/analysis
python run_dataset_analysis.py \
    --dataset task_classification_datasets/rerange+spatial_matched_subtasks.json.gz
```

---

## Top 10 Episodes for Visualization

| Episode ID | Index | Task Summary | Recommended Viz |
|------------|-------|--------------|-----------------|
| 1312 | 64 | plate, bowl → dining table | Video |
| 1317 | 65 | picture frame, vase → dining room | Video |
| 1378 | 67 | laptop stand, monitor stand → living room | Video |
| 1327 | 66 | laptop, laptop stand → bedroom | Video |
| 1425 | 69 | toy airplane, helmet → bedroom | Video |
| 212 | 15 | tomato, sushi mat → dining room | Video |
| 214 | 17 | 4 kitchen items → living room | Video |
| 866 | 38 | plate, bowl → kitchen | Video |
| 1471 | 71 | vase, candle → dining table | Video |
| 153 | 3 | toy bee, construction set → closet | Video |

---

## File Locations

```
partnr-planner/
├── scripts/
│   ├── prediviz/              # PrediViz static diagrams
│   │   ├── viz.py
│   │   ├── conf/config.yaml
│   │   └── entities/          # Visualization components
│   └── hitl_analysis/
│       └── visualize_episode.py
├── dataset_generation/
│   └── benchmark_generation/
│       └── analysis/          # Dataset analysis plots
│           ├── run_dataset_analysis.py
│           └── task_type_upset_plot.py
├── habitat_llm/
│   ├── evaluation/
│   │   └── evaluation_runner.py  # Video & top-down viz
│   └── examples/
│       └── example_utils.py      # DebugVideoUtil class
└── outputs/
    └── habitat_llm/           # All visualization outputs
        └── <run>/
            └── results/
                ├── videos/    # Recorded videos
                └── traces/    # Text traces
```
