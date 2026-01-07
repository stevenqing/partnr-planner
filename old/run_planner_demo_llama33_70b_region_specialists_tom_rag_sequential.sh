#!/bin/bash

# 使用 OpenRouter + Llama 3.3 70B Instruct 的区域专家组合脚本 (ToM + RAG 版本)
# - Agent 0: Kitchen Specialist (厨房专家) - 只能访问厨房 + Theory of Mind
# - Agent 1: Living Room Specialist (客厅专家) - 只能访问客厅 + Theory of Mind
# - 使用 Theory of Mind (ToM) prompt 结合区域限制
# - 启用 RAG 功能
# - 顺序执行模式（sequential execution）
# - 通过 openai_chat 后端，用 OpenRouter 的 OpenAI 兼容接口调用 Llama 3.3 70B

#============ 配置 OpenRouter ============#
export OPENAI_API_KEY="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"
export OPENAI_ENDPOINT="https://openrouter.ai/api/v1"

#============ GPU / EGL 配置 ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ 运行 planner demo ============#
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    +evaluation.sequential_execution=True \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model="meta-llama/llama-3.3-70b-instruct:free" \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.model="meta-llama/llama-3.3-70b-instruct:free" \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=agent_rules_kitchen_only_tom \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=agent_rules_living_room_specialist_tom \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/rag_datasets/spatial_rerange_merged/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/rag_datasets/spatial_rerange_merged/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=react \
    evaluation.agents.agent_1.planner.plan_config.example_type=react
