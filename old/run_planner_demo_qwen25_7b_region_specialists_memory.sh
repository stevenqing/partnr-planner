#!/bin/bash

# 使用 Qwen2.5-7B-Instruct 的区域专家组合脚本 (ToM + Memory 版本)
# - Agent 0: Kitchen Specialist (厨房专家) - 只能访问厨房 + Theory of Mind
# - Agent 1: Living Room Specialist (客厅专家) - 只能访问客厅 + Theory of Mind
# - 使用 Theory of Mind (ToM) prompt 结合区域限制
# - 启用 Memory 功能 (MEMENTO) - 场景特定的示例检索
# - 顺序执行模式（sequential execution）
# - 使用本地 Qwen2.5-7B-Instruct 模型（HuggingFace）

#============ GPU / EGL 配置 ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ 运行 planner demo ============#
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    ++evaluation.sequential_execution=True \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine="Qwen/Qwen2.5-7B-Instruct" \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine="Qwen/Qwen2.5-7B-Instruct" \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=agent_rules_kitchen_only_tom \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=agent_rules_living_room_specialist_tom \
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
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=5
