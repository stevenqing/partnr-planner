#!/bin/bash

# Ours Method (Hierarchical Memory): Eval H+R, Memory H+R
# Test on Heterogeneous+Rearrange tasks, retrieve from H+R hierarchical memory
#
# Uses hierarchical 4-stage retrieval pipeline with L_ind and L_coop skills

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Python Path Configuration ============#
export PYTHONPATH=/home/a5l/shuqing.a5l/partnr-planner:$PYTHONPATH

#============ Run planner demo ============#
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    num_proc=14 \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+rerange.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory_rule/hierarchical_heterogeneous_rerange/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory_rule/hierarchical_heterogeneous_rerange/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[hierarchical_memory] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[hierarchical_memory] \
    evaluation.agents.agent_0.planner.plan_config.example_type=hierarchical \
    evaluation.agents.agent_1.planner.plan_config.example_type=hierarchical \
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.include_coop_skills=True \
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.include_coop_skills=True \
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.include_ind_skills=True \
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.include_ind_skills=True \
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.use_hierarchical_structure=True \
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.use_hierarchical_structure=True \
    +evaluation.agents.agent_0.planner.plan_config.ablation_config.random_retrieval=False \
    +evaluation.agents.agent_1.planner.plan_config.ablation_config.random_retrieval=False \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills
