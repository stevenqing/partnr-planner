#!/bin/bash

# Ours Method (Hierarchical Memory): Eval H+R+S+T, Memory H+R+S+T
# Test on Heterogeneous+Rearrange+Spatial+Temporal tasks, retrieve from all 4 hierarchical memories:
#   - hierarchical_heterogeneous_rerange (H): 280 ind + 175 coop skills
#   - hierarchical_heterogeneous_temporal (T): 7 ind + 5 coop skills
#   - hierarchical_rerange_only (R): 223 ind + 227 coop skills
#   - hierarchical_spatial_only (S): 14 ind + 6 coop skills
#
# Uses hierarchical 4-stage retrieval pipeline with L_ind and L_coop skills

#============ GPU / EGL Configuration ============#
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Python Path Configuration ============#
export PYTHONPATH=/home/a5l/shuqing.a5l/partnr-planner:$PYTHONPATH

#============ Run planner demo ============#
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    num_proc=5 \
    habitat.dataset.data_path="task_classification_datasets/heterogeneous+rerange+spatial+temporal.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=2000 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=2000 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
    +evaluation.agents.agent_0.planner.plan_config.rag_top_k=5 \
    +evaluation.agents.agent_1.planner.plan_config.rag_top_k=5 \
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_heterogeneous_rerange/,data/hierarchical_skill_memory/hierarchical_heterogeneous_temporal/,data/hierarchical_skill_memory/hierarchical_rerange_only/,data/hierarchical_skill_memory/hierarchical_spatial_only/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_heterogeneous_rerange/,data/hierarchical_skill_memory/hierarchical_heterogeneous_temporal/,data/hierarchical_skill_memory/hierarchical_rerange_only/,data/hierarchical_skill_memory/hierarchical_spatial_only/] \
    evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[hierarchical_memory,hierarchical_memory,hierarchical_memory,hierarchical_memory] \
    evaluation.agents.agent_1.planner.plan_config.rag_data_source_name=[hierarchical_memory,hierarchical_memory,hierarchical_memory,hierarchical_memory] \
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
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4 \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4
