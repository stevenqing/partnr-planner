#!/bin/bash

# 运行 planner demo 的 bash 脚本 - 启用 RAG 功能 + Cooperation Skills
# 使用 Cooperation Skills 数据集和 Qwen2.5-7B-Instruct 模型
# 使用 rag_prompt_sequential_cooperation_skills_qwen prompt (ToM + Cooperation Skills)

#============ GPU / EGL 配置 ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ 运行 planner demo ============#
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=2500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=2500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine="Qwen/Qwen2.5-7B-Instruct" \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine="Qwen/Qwen2.5-7B-Instruct" \
    evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/rag_datasets/cooperation_skills_org_re_sp/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/rag_datasets/cooperation_skills_org_re_sp/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=react \
    evaluation.agents.agent_1.planner.plan_config.example_type=react \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_qwen \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_qwen
