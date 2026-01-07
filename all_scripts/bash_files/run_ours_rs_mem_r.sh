#!/bin/bash

# Ours Method (Hierarchical Memory): Eval R+S, Memory R only
# Test on Rearrange+Spatial tasks, retrieve from R only hierarchical memory
#
# Uses hierarchical 4-stage retrieval pipeline:
# 1. Query Generation: q_t = f_query(w_t^self, w_t^env, delta_w_{t-1}, g)
# 2. Abstract Skill Matching: S_candidate = {s in L : sim(q_t, name(s)) > theta}
# 3. Instance Retrieval: I_retrieved = union top-k{sim(context(i), w_t)}
# 4. Executability Filtering
#
# Ablation config (set in script):
# - include_coop_skills: True (include L_coop)
# - include_ind_skills: True (include L_ind)
# - use_hierarchical_structure: True (use hierarchical retrieval)
# - random_retrieval: False (use similarity-based retrieval)

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
    num_proc=4 \
    habitat.dataset.data_path="task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
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
    evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_rerange_only/] \
    evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/hierarchical_skill_memory/hierarchical_rerange_only/] \
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
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4 \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills_v4
