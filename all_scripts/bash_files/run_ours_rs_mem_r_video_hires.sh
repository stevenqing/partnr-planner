#!/bin/bash

# Ours Method (Hierarchical Memory): Eval R+S, Memory R only - HIGH-RES VIDEO RECORDING
# Test on Rearrange+Spatial tasks, retrieve from R only hierarchical memory
#
# This script enables HIGH-RESOLUTION video recording (1024x1024) for visualization.
# Set EPISODE_INDICES to run specific episodes, or leave empty for all episodes.
#
# Usage:
#   bash run_ours_rs_mem_r_video_hires.sh                    # Run all episodes
#   EPISODE_INDICES="[0,1,2]" bash run_ours_rs_mem_r_video_hires.sh  # Run specific episodes

#============ Video Recording Configuration ============#
# Set to specific episode indices like "[0,1,2]" or leave empty for all
# Top 10 most reliable successful episodes (episode IDs -> indices):
# 1312->64, 1317->65, 1327->66, 1378->67, 1425->69, 212->15, 214->17, 866->38, 1471->71, 153->3
EPISODE_INDICES=${EPISODE_INDICES:-"[64,65,66,67,69,15,17,38,71,3]"}  # Default: top 10 most reliable episodes

# Number of processes (recommend 1 for video recording to avoid conflicts)
NUM_PROC=${NUM_PROC:-1}

# Video resolution (default: 1024x1024 for high-res)
VIDEO_RES=${VIDEO_RES:-1024}

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Python Path Configuration ============#
export PYTHONPATH=/home/a5l/shuqing.a5l/partnr-planner:$PYTHONPATH

#============ Build command arguments ============#
CMD_ARGS=(
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml
    evaluation.sequential_execution=True
    evaluation.save_video=True
    num_proc=${NUM_PROC}
    habitat.dataset.data_path="task_classification_datasets/rerange+spatial_matched_subtasks.json.gz"
    # High-resolution video settings for agent_0 (third_rgb for video)
    habitat.simulator.agents.agent_0.sim_sensors.third_rgb_sensor.height=${VIDEO_RES}
    habitat.simulator.agents.agent_0.sim_sensors.third_rgb_sensor.width=${VIDEO_RES}
    # High-resolution video settings for agent_1 (third_rgb for video)
    habitat.simulator.agents.agent_1.sim_sensors.third_rgb_sensor.height=${VIDEO_RES}
    habitat.simulator.agents.agent_1.sim_sensors.third_rgb_sensor.width=${VIDEO_RES}
    # High-resolution picture settings for agent_0 (head_rgb for task_viz images)
    habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.height=${VIDEO_RES}
    habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.width=${VIDEO_RES}
    # High-resolution picture settings for agent_1 (head_rgb for task_viz images)
    habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.height=${VIDEO_RES}
    habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.width=${VIDEO_RES}
    # LLM configuration
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct
    # RAG configuration
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True
    +evaluation.agents.agent_0.planner.plan_config.rag_top_k=5
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=5
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_rerange_only/]
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_rerange_only/]
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[hierarchical_memory]
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[hierarchical_memory]
    evaluation.agents.agent_0.planner.plan_config.example_type=hierarchical
    evaluation.agents.agent_1.planner.plan_config.example_type=hierarchical
    # Ablation configuration
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.include_coop_skills=True
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.include_coop_skills=True
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.include_ind_skills=True
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.include_ind_skills=True
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.use_hierarchical_structure=True
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.use_hierarchical_structure=True
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.random_retrieval=False
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.random_retrieval=False
    # Instruction configuration
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4
)

# Add episode indices if specified
if [ -n "${EPISODE_INDICES}" ]; then
    CMD_ARGS+=("+episode_indices=${EPISODE_INDICES}")
    echo "Running with episode indices: ${EPISODE_INDICES}"
fi

echo "============================================"
echo "Running with HIGH-RES VIDEO RECORDING ENABLED"
echo "Video resolution: ${VIDEO_RES}x${VIDEO_RES}"
echo "Number of processes: ${NUM_PROC}"
echo "============================================"

#============ Run planner demo ============#
python -m habitat_llm.examples.planner_demo "${CMD_ARGS[@]}"
