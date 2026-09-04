# Decentralized Skill-Structured Hierarchical Memory for Multi-Agent Coordination

This folder contains the implementation of the hierarchical memory framework for multi-agent coordination.

## Overview

The framework enables decentralized multi-agent coordination through a two-stage approach:

1. **Memory Building Stage**: Distill successful trajectories into a hierarchical memory system
2. **Memory Utilization Stage**: Coordinate execution through hierarchical retrieval and sequential observation

## Files

| File | Description |
|------|-------------|
| `build_hierarchical_skill_memory.py` | Core memory building: extracts L_ind and L_coop skills |
| `hierarchical_retrieval.py` | Four-stage hierarchical retrieval pipeline |
| `theory_of_mind.py` | Theory of Mind reasoning for partner intention inference |
| `build_rag_dataset.py` | Build RAG dataset from heuristic trajectories |
| `planner_integration.py` | Integration with LLM planner infrastructure |
| `run_build_memory.py` | Main script to build memory and dataset |

## Quick Start

### 1. Build Memory from Heuristic Results

```bash
python our_method/run_build_memory.py \
    --results-dir /path/to/heuristic/results \
    --memory-output-dir /path/to/save/memory \
    --rag-output-dir /path/to/save/rag_dataset
```

### 2. Use in Python

```python
from our_method import create_planner, create_rag_integration

# Create planner with hierarchical memory
planner = create_planner('/path/to/memory')

# Get action recommendation
result = planner.step(
    world_state={'objects': {...}, 'furniture': {...}},
    goal='Move the cup to the kitchen',
    partner_action='Navigate[plate_0]'
)

# Or use RAG integration
rag = create_rag_integration('/path/to/memory', '/path/to/rag_dataset')
examples = rag.get_examples(instruction, world_state)
```

## Architecture

### Skill-Structured Hierarchical Memory

```
L = L_ind ∪ L_coop

L_ind: Individual skills (single-agent competencies)
L_coop: Cooperation skills (partner-aware conditional policies)

Skill s = (name, type, I)
Instance i = (context, demo, e_src)
```

### Hierarchical Retrieval Pipeline

```
Stage 1: Query Generation
    q_t = f_query(w_t^self, w_t^env, Δw_{t-1}, g)

Stage 2: Abstract Skill Matching
    S_candidate = {s ∈ L : sim(q_t, name(s)) > θ_abstract}

Stage 3: Instance Retrieval
    I_retrieved = ∪_{s ∈ S_candidate} top-k{sim(context(i), w_t)}

Stage 4: Executability Filtering
    S_executable = {s : precond_s(w_t^self) = true}
```

### Theory of Mind Reasoning

```
1. Belief Formation: Analyze state changes to understand partner's knowledge
2. Hypothesis Generation: Predict partner's goals/strategy
3. Prediction & Planning: Plan coordinated action
```

## Reference

See `/doc/our_method` for the full theoretical framework.
