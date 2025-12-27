#!/bin/bash

# 使用 Llama 3.1-8B 的区域专家组合脚本 (Org 版本)
# - Agent 0: Kitchen Specialist (厨房专家) - 只能访问厨房
# - Agent 1: Living Room Specialist (客厅专家) - 只能访问客厅
# - 使用最简单的 zero-shot prompt (zero_shot_prompt_org)
# - 禁用 RAG 功能
# - 顺序执行模式（sequential execution）
# - 使用本地 Llama 3.1-8B 模型（HuggingFace）

#============ GPU / EGL 配置 ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ 运行 planner demo ============#
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    +evaluation.sequential_execution=True \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=llama \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=llama \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine="meta-llama/Llama-3.1-8B-Instruct" \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine="meta-llama/Llama-3.1-8B-Instruct" \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=agent_rules_kitchen_only \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=agent_rules_living_room_specialist \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=False \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=False
