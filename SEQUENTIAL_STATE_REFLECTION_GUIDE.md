# Sequential Execution with State Reflection - Implementation Guide

## Overview

This guide describes enhancements to the sequential execution mode that enable agents to **infer other agents' actions** by comparing state changes, rather than directly observing their actions.

## Key Concept

In sequential execution:
- **Agent 0** acts first with initial observations
- **Agent 1** acts second and sees the **updated environment state** (not Agent 0's actions directly)

The improvement: Agent 1 can **deduce what Agent 0 did** by comparing the current state with the previous state.

## Implementation Components

### 1. State Tracking in Planner (`llm_planner.py`)

**Added attributes:**
```python
# Track previous world state for state comparison
self.previous_world_description: str = ""
self.previous_obj_states: str = ""
```

**State comparison in prompt building:**
- Automatically computes differences between previous and current object states
- Provides formatted state change summary to the agent
- Uses `{state_comparison}` placeholder in prompts

### 2. State Difference Computation (`utils.py`)

**Function: `compute_state_differences(previous_obj_descr, current_obj_descr)`**

Features:
- Parses object descriptions to identify changes
- Detects:
  - Objects that moved locations
  - Objects newly discovered
  - Objects picked up by other agent
  - Objects placed by other agent
- Provides interpretation hints:
  - "Other agent is currently holding an object"
  - "Other agent likely performed pick and/or place actions"
  - "Other agent likely explored a new area"

Example output:
```
State Changes Detected:
  • apple: MOVED from 'kitchen_counter' to 'held by other agent'
  • plate: NEWLY DISCOVERED at 'cabinet_0'

Interpretation Hints:
  → Other agent is currently holding an object
  → Other agent likely performed pick and/or place actions
  → Other agent likely explored a new area or opened containers
```

### 3. Enhanced Prompt Template

**File: `rag_prompt_sequential_state_reflection.yaml`**

Key sections:

#### A. State Observation Instructions
```yaml
CRITICAL REASONING PROCESS - You MUST follow these steps in your Thought:

1. **Compare Current State with Previous State**:
   - What objects have moved locations since your last observation?
   - What objects are now held by "the other agent" that weren't before?
   ...

2. **Infer Other Agent's Actions**:
   - Based on the state differences, what actions did the other agent likely perform?
   ...

3. **Understand Task Division**:
   - Based on what the other agent did, what parts of the task are completed?
   ...

4. **Make Strategic Decision**:
   - Given what the other agent accomplished, what is the most effective next action?
   ...
```

#### B. State Comparison Integration
```yaml
{world_description}

## Previous State Analysis
{state_comparison}
```

The `{state_comparison}` placeholder is automatically filled with the output from `compute_state_differences()`.

#### C. Example Reasoning Format
```yaml
Example format with proper state reflection:
Thought: [State Comparison] Comparing current state with previous:
(1) apple is now "held by other agent" instead of being on kitchen_counter,
(2) dining_table now shows available space.
[Other Agent Inference] The other agent likely performed Navigate[kitchen_counter]
and Pick[apple], preparing to move it.
[Task Progress Analysis] The task requires moving apple to dining_table.
Other agent has completed the pick phase but hasn't placed yet.
[Strategic Decision] Based on 'coordination_handoff' skill pattern from examples,
when other agent is mid-action on an object, I should work on a different subtask.
I'll focus on finding the plate mentioned in the task.
Explore[kitchen]
Assigned!
```

## Usage

### Option 1: Use State Reflection Prompt

Update your script to use the new prompt:

```bash
python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \
    instruct@evaluation.agents.agent_0.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    instruct@evaluation.agents.agent_1.planner.plan_config.instruct=rag_prompt_sequential_state_reflection \
    # ... other parameters
```

### Option 2: Update Your Script

Modify `run_planner_demo_with_rag.sh`:

```bash
#!/bin/bash

python -m habitat_llm.examples.planner_demo \
    --config-name baselines/decentralized_zero_shot_react_summary_with_rag.yaml \
    evaluation.sequential_execution=True \  # Enable sequential mode
    habitat.dataset.data_path="/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=800 \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=800 \
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
```

## Benefits

### For Agent 0 (First Actor)
- Understands its role as laying the foundation
- Focuses on exploration and object positioning
- Knows Agent 1 will see and build upon its work

### For Agent 1 (Second Actor)
- **Automatic state comparison** shows exactly what changed
- **Inferred actions** help understand Agent 0's strategy
- **Task progress analysis** identifies what's completed vs. remaining
- **Strategic coordination** based on Agent 0's choices

### Overall Improvements
1. **Better coordination**: Agents understand implicit task division
2. **Reduced redundancy**: Agent 1 avoids repeating Agent 0's work
3. **Complementary actions**: Agent 1 completes tasks Agent 0 cannot (Clean, Fill, PowerOn/Off)
4. **Explicit reasoning**: Forced structured thinking improves decision quality
5. **RAG integration**: State reflection combined with skill patterns from successful examples

## Example Scenario

**Task**: "Move the apple to the dining table and clean it"

**Agent 0's turn:**
```
Thought: [Initial state] No previous observations. Task requires finding apple and moving it.
As Agent 0, I should focus on finding and positioning the apple. Agent 1 will handle cleaning.
Explore[kitchen]
```

**After Agent 0 acts:**
- Apple discovered at kitchen_counter
- Apple picked up and placed on dining_table

**Agent 1's turn with state reflection:**
```
State Changes Detected:
  • apple: MOVED from 'kitchen_counter' to 'dining_table'

Interpretation Hints:
  → Other agent likely performed pick and/or place actions

Thought: [State Comparison] The apple was at kitchen_counter before, now it's at dining_table.
[Other Agent Inference] Agent 0 successfully navigated to kitchen, picked up the apple,
and placed it on the dining_table.
[Task Progress Analysis] Task is 50% complete - apple is positioned correctly but not cleaned.
[Strategic Decision] As Agent 1, I have cleaning capability that Agent 0 lacks.
I should now clean the apple that Agent 0 positioned.
Navigate[dining_table]
```

Then:
```
Thought: [State Comparison] No new changes, apple still on dining_table.
[Strategic Decision] I'm at the dining table now, time to clean the apple.
Clean[apple]
```

## Comparison with Original Approach

| Aspect | Original | With State Reflection |
|--------|----------|----------------------|
| Agent 1's awareness | Sees final state only | Sees state + changes + interpretation |
| Action inference | Implicit/unclear | Explicit in prompt |
| Reasoning structure | Freeform | Structured 4-step process |
| Coordination | Ad-hoc | Systematic based on state analysis |
| Debugging | Hard to understand agent reasoning | Clear reasoning trace |

## Future Enhancements

Potential improvements:
1. **Action history integration**: Combine state changes with actual action history when available
2. **Confidence scoring**: Rate how certain the inference is about other agent's actions
3. **Multi-step inference**: Track changes across multiple steps to understand sequences
4. **Failure detection**: Identify when other agent's actions failed based on unexpected state
5. **Proactive coordination**: Suggest complementary actions based on inferred strategy

## Notes

- State comparison works best with `sequential_execution=True`
- Works in parallel mode too, but less effective (no guaranteed ordering)
- Compatible with existing RAG and skill pattern systems
- Minimal performance overhead (simple string comparison)
