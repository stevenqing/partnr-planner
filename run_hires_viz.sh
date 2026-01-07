#!/bin/bash

# High-resolution visualization run for Episode 64 (episode_id 1312)
# Output resolution: 1024x1024 for video

#============ GPU / EGL Configuration ============#
export CUDA_VISIBLE_DEVICES=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

#============ Conda Environment ============#
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

cd /home/a5l/shuqing.a5l/partnr-planner

#============ Run planner demo with high-res video ============#
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    evaluation.save_video=True \
    num_proc=1 \
    habitat.dataset.data_path="task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    +episode_indices=[64] \
    habitat.simulator.agents.agent_0.sim_sensors.third_rgb_sensor.height=1024 \
    habitat.simulator.agents.agent_0.sim_sensors.third_rgb_sensor.width=1024 \
    habitat.simulator.agents.agent_1.sim_sensors.third_rgb_sensor.height=1024 \
    habitat.simulator.agents.agent_1.sim_sensors.third_rgb_sensor.width=1024 \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=1500 \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=/home/a5l/shuqing.a5l/models/Llama-3.1-8B-Instruct \
    evaluation.agents.agent_0.planner.plan_config.enable_rag=True \
    evaluation.agents.agent_1.planner.plan_config.enable_rag=True \
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
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_cooperation_skills
