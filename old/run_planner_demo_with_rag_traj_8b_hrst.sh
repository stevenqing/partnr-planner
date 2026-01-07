#!/bin/bash

# 运行 planner demo 的 bash 脚本 - 启用RAG功能
# 使用前五个任务的数据集和 Meta-Llama-3-8B-Instruct 模型

python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1200 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/rag_datasets/rerange_only_organized_by_skills/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/rag_datasets/rerange_only_organized_by_skills/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[react_trajectories] \
    evaluation.agents.agent_0.planner.plan_config.example_type=skills \
    evaluation.agents.agent_1.planner.plan_config.example_type=skills \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_state_reflection
    # rag_prompt_with_skill_patterns
    # zero_shot_prompt_with_rag
    # rerange_only_cleaned
    # rag_prompt_with_skill_patterns
    # evaluation.agents.agent_0.planner.plan_config.example_type=react \
    # evaluation.agents.agent_1.planner.plan_config.example_type=react \
