#!/bin/bash

# Traj-based Method: Transfer R+H+S+T, Memory T only
# Test on Rearrange+Hetero+Spatial+Temporal tasks, retrieve from Temporal only trajectories

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Run planner demo ============#
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/rag_datasets/react_rag_dataset_dedup_v2/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/rag_datasets/react_rag_dataset_dedup_v2/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=react \
    evaluation.agents.agent_1.planner.plan_config.example_type=react \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_state_reflection
