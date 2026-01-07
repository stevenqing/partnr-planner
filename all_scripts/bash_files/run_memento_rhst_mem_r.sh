#!/bin/bash

# MEMENTO Method: Transfer R+H+S+T, Memory R only
# Test on Rearrange+Hetero+Spatial+Temporal tasks, retrieve from Rearrange only memory

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Run planner demo ============#
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    num_proc=5 \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/memory_memento_dataset/rerange_only/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/memory_memento_dataset/rerange_only/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=memento \
    evaluation.agents.agent_1.planner.plan_config.example_type=memento \
    +evaluation.agents.agent_0.planner.plan_config.memory_path=data/memory_memento_dataset/rerange_only \
    +evaluation.agents.agent_1.planner.plan_config.memory_path=data/memory_memento_dataset/rerange_only \
    +evaluation.agents.agent_0.planner.plan_config.ensure_same_scene=False \
    +evaluation.agents.agent_1.planner.plan_config.ensure_same_scene=False \
    +evaluation.agents.agent_0.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_1.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_0.planner.plan_config.rag_top_k=3 \
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=3 \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_enhanced \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_enhanced
