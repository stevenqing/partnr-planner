#!/bin/bash

# 运行 planner demo 的 bash 脚本
# 使用前五个任务的数据集和 Meta-Llama-3-8B-Instruct 模型

python -m habitat_llm.examples.planner_demo --config-name baselines/heuristic_full_obs.yaml \
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/heterogeneous+temporal.json.gz"

# python -m enhanced_planner_demo \
#     --config-name simple_enhanced_config.yaml \
#     habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets_first5/rerange+spatial_first5.json.gz" \
#     evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
#     evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
#     evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct \
#     evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
