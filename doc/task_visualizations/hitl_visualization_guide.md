# HITL Visualization Guide

## Overview

HITL (Human-In-The-Loop) visualization creates animated videos of episode rollouts showing:
- Agent positions and movements
- Object tracking with color coding
- Task completion metrics
- Grabbed objects overlay
- Action event timeline

## Data Requirements

HITL visualization requires **specific data format** from human data collection sessions:

```
hitl_data/
├── raw/
│   └── [session_name]/
│       ├── [episode_id].json.gz   # Per-episode execution data
│       └── session.json.gz        # Session metadata
└── processed/
    ├── best/                      # Successful episodes
    └── failed/                    # Failed episodes
```

### Required Fields in Episode JSON
- `frames`: List of timestamped simulation states
- `episode`: Episode metadata
- `session`: Session configuration
- Agent states, object states per frame

---

## Available HITL Data

You have pre-collected HITL data in tar.gz format:

```bash
ls data/datasets/partnr_episodes/hitl_data/
# p5_learn_val_finetune.tar.gz
# p5_learn_val_v2.tar.gz
# p5_multi_val.tar.gz
# p5_single_train_2k.tar.gz
# p5_single_val.tar.gz
```

---

## Step-by-Step: Generate HITL Visualizations

### Step 1: Extract HITL Data

```bash
cd /home/a5l/shuqing.a5l/partnr-planner

# Create extraction directory
mkdir -p data/hitl_data/p5_single_val

# Extract data
tar -xzf data/datasets/partnr_episodes/hitl_data/p5_single_val.tar.gz \
    -C data/hitl_data/p5_single_val
```

### Step 2: Preprocess Data

```bash
python scripts/hitl_analysis/preprocess_data.py \
    --collection-path data/hitl_data/p5_single_val
```

This creates:
- `processed/best/` - Successful episode files
- `processed/failed/` - Failed episode files
- `processed_metrics.json` - Aggregated metrics

### Step 3: Generate Videos

```bash
python scripts/hitl_analysis/visualize_episode.py \
    --episodes-path data/hitl_data/p5_single_val/processed/best \
    --dataset-file data/versioned_data/partnr_episodes/v0_0/val.json.gz \
    --ncpus 4 \
    --multi \
    --multi-plot
```

### Output
Videos saved to: `data/hitl_data/p5_single_val/processed/best/videos/`

---

## Can I Use This With My LLM Planner Results?

**Short answer: No, directly.**

Your current data (`outputs/habitat_llm/`) uses a different format:
- `detailed_traces/*.pkl` - Pickled action/state history
- `traces/*.txt` - Text-based action logs

### Why They're Different

| HITL Data | LLM Planner Data |
|-----------|------------------|
| Records every simulation frame | Records high-level actions |
| Contains camera sensor data | No visual data stored |
| Human interaction timestamps | LLM response timestamps |
| `frames[].agent_states.position` | Action history only |

### Alternative: Use Video Recording

For your LLM planner results, use **evaluation video recording** instead:

```bash
# This is what job 167990 is doing
python -m habitat_llm.examples.planner_demo \
    evaluation.save_video=True \
    episode_indices=[64,65,66,67,69] \
    num_proc=1
```

---

## Quick Reference: HITL Visualization Commands

```bash
# 1. Extract (choose one dataset)
tar -xzf data/datasets/partnr_episodes/hitl_data/p5_single_val.tar.gz \
    -C data/hitl_data/p5_single_val

# 2. Preprocess
python scripts/hitl_analysis/preprocess_data.py \
    --collection-path data/hitl_data/p5_single_val

# 3. Visualize
python scripts/hitl_analysis/visualize_episode.py \
    --episodes-path data/hitl_data/p5_single_val/processed/best \
    --dataset-file data/versioned_data/partnr_episodes/v0_0/val.json.gz \
    --ncpus 4 --multi --multi-plot
```

---

## Available HITL Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `p5_single_val.tar.gz` | Single-agent validation | 51MB |
| `p5_single_train_2k.tar.gz` | Single-agent training (2k) | 61MB |
| `p5_multi_val.tar.gz` | Multi-agent validation | 54MB |
| `p5_learn_val_v2.tar.gz` | Learning validation v2 | 271MB |
| `p5_learn_val_finetune.tar.gz` | Finetuning validation | 254MB |

---

## Summary

| Your Data Type | Visualization Method |
|----------------|---------------------|
| LLM Planner outputs (`outputs/habitat_llm/`) | Video Recording (`save_video=True`) ✅ |
| HITL collected data (`hitl_data/*.tar.gz`) | HITL Visualization ✅ |
| Custom datasets | Video Recording ✅ |
