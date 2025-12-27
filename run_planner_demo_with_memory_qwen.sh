#!/bin/bash

# 运行 planner demo 的 bash 脚本 - 使用Qwen模型并启用Memory功能 (MEMENTO)
# 使用 MEMENTO 内存系统进行场景特定的示例检索
# 使用 Qwen2.5-7B-Instruct 模型

# Configure headless EGL rendering with NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Use conda python directly
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+heterogeneous_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[/home/shuqing/partnr-planner/memory_from_trajectories_success/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[/home/shuqing/partnr-planner/memory_from_trajectories_success/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[habitat_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=react \
    evaluation.agents.agent_1.planner.plan_config.example_type=react \
    +evaluation.agents.agent_0.planner.plan_config.memory_path=habitat_trajectories/trajectory_memory \
    +evaluation.agents.agent_1.planner.plan_config.memory_path=habitat_trajectories/trajectory_memory \
    +evaluation.agents.agent_0.planner.plan_config.ensure_same_scene=True \
    +evaluation.agents.agent_1.planner.plan_config.ensure_same_scene=True \
    +evaluation.agents.agent_0.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_1.planner.plan_config.corresponding_memory=False \
    +evaluation.agents.agent_0.planner.plan_config.rag_top_k=5 \
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=5 \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=memory_prompt_qwen \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=memory_prompt_qwen
