#!/bin/bash

# 运行 planner demo 的 bash 脚本
# 使用前五个任务的数据集和 Meta-Llama-3-8B-Instruct 模型

# Configure headless EGL rendering with NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Use conda python directly
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=llama \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=llama \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine="meta-llama/Llama-3.1-8B-Instruct" \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine="meta-llama/Llama-3.1-8B-Instruct" \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt_tom \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt_tom \
