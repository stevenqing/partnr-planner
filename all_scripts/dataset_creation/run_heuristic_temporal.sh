#!/bin/bash

# Generate trajectories for Temporal tasks using heuristic planner
# Output will be used to create MEMENTO memory for T only

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "=== Generating trajectories for Temporal tasks ==="

python -m habitat_llm.examples.planner_demo \
    --config-name baselines/heuristic_full_obs.yaml \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+temporal.json.gz"
