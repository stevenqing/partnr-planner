# Sequential Execution Mode Guide

## Overview

The sequential execution mode allows you to control whether agents in multi-agent tasks act in parallel or sequentially. This is particularly useful for ensuring that agent_1 can explicitly see the effects of agent_0's actions.

## Execution Modes

### Parallel Mode (Default)
- **Behavior**: All agents plan and act simultaneously with the same observations
- **Use case**: Standard multi-agent coordination where agents need to make independent decisions
- **Parameter**: `evaluation.sequential_execution=False`

### Sequential Mode
- **Behavior**: Agents act one at a time in order (agent_0 → agent_1), with each subsequent agent seeing the updated environment state after previous agents have acted
- **Use case**: When you want explicit agent ordering and later agents to observe earlier agents' effects
- **Parameter**: `evaluation.sequential_execution=True`

## How to Enable Sequential Execution

### Method 1: Via Configuration File

Edit your baseline config file (e.g., `habitat_llm/conf/baselines/decentralized_zero_shot_react_summary_with_rag.yaml`):

```yaml
evaluation:
  sequential_execution: True  # Enable sequential execution
  agents:
    agent_0:
      # ... agent_0 config
    agent_1:
      # ... agent_1 config
```

### Method 2: Via Command Line Override

When running your script, add the parameter as a command-line argument:

```bash
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    # ... other parameters
```

### Method 3: In Your Shell Script

Update your shell script to include the parameter:

```bash
# For parallel execution (default)
evaluation.sequential_execution=False \

# For sequential execution
evaluation.sequential_execution=True \
```

## Example: Updated run_planner_demo_rag_qwen.sh

```bash
#!/bin/bash

# Configure headless EGL rendering with NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0,1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Use conda python directly
/home/shuqing/.conda/envs/habitat-llm/bin/python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \  # ENABLE SEQUENTIAL EXECUTION HERE
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen \
    # ... rest of parameters
```

## Implementation Details

### Sequential Execution Flow

When `sequential_execution=True`:

1. **Agent 0 plans** using initial observations
2. **Agent 0 acts** and the environment is updated
3. **Agent 1 observes** the updated environment (including agent 0's effects)
4. **Agent 1 plans** using the updated observations
5. **Agent 1 acts** and the environment is updated again

### Code Changes

The implementation is in `habitat_llm/evaluation/decentralized_evaluation_runner.py`:

- `get_low_level_actions()`: Routes to parallel or sequential mode based on config
- `get_low_level_actions_parallel()`: Original parallel execution logic
- `get_low_level_actions_sequential()`: New sequential execution logic

## Use Cases

### When to Use Sequential Execution

1. **Agent coordination studies**: Research on how agents respond to each other's actions
2. **Turn-based collaboration**: Tasks where agents naturally take turns
3. **Hierarchical control**: When one agent should act as a "leader" and another as a "follower"
4. **Debugging**: Understanding exact agent interaction sequences

### When to Use Parallel Execution

1. **Standard multi-agent tasks**: Default collaborative task solving
2. **Independent agents**: When agents work on separate subtasks
3. **Simultaneous actions**: Tasks requiring concurrent agent actions
4. **Baseline comparisons**: Matching original experiment setups

## Performance Considerations

- **Sequential mode**: May be slightly slower due to multiple environment steps per planning cycle
- **Parallel mode**: Faster execution but agents don't see immediate effects of other agents
- Both modes produce the same final actions dictionary for logging purposes

## Testing

To test the feature:

```bash
# Test parallel mode
bash run_planner_demo_rag_qwen.sh  # Uses sequential_execution=False by default

# Test sequential mode
# Edit run_planner_demo_rag_qwen.sh and change to evaluation.sequential_execution=True
bash run_planner_demo_rag_qwen.sh
```

Compare the agent behaviors and task completion metrics between the two modes.
