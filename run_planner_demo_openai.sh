#!/bin/bash

# 运行 planner demo 的 bash 脚本 - 使用 OpenAI GPT 模型
# 使用前五个任务的数据集和 OpenAI GPT 模型

# 设置 OpenAI API Key 和 Endpoint
# 如果使用标准 OpenAI API，设置 OPENAI_ENDPOINT=api.openai.com
# 如果使用 Azure OpenAI，设置 OPENAI_ENDPOINT 为你的 Azure endpoint
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
export OPENAI_ENDPOINT="api.openai.com"  # 标准 OpenAI API，如果是 Azure 请修改为你的 endpoint

# Configure headless EGL rendering with NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Use conda python directly
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/heterogeneous+temporal.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.model=gpt-4o \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.model=gpt-4o \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt_gpt \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt_gpt \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False
