# MEMENTO: Memory-Enhanced Multi-Agent Personalized Assistance

Implementation of the MEMENTO framework for personalized embodied agent assistance.

Based on: "Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization" (Kwon et al., 2025)

## Overview

MEMENTO addresses the challenge of personalized assistance in embodied AI by implementing a hierarchical knowledge graph-based user profile memory. This enables agents to:

- **Understand personalized knowledge**: Interpret user-specific object semantics
- **Remember user patterns**: Leverage behavioral patterns from past interactions
- **Ground task planning**: Use episodic memory for personalized goal derivation

## Architecture

### Three-Level Hierarchical Structure

```
┌─────────────────────────────────────────────────────────────┐
│                         USER LEVEL                          │
│                    (user_0, user_1, ...)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │ refers_to
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      KNOWLEDGE LEVEL                        │
│     Object Semantics          │        User Patterns        │
│   ("my coffee set")           │   ("my morning routine")    │
└──────────┬────────────────────┴──────────────┬──────────────┘
           │ composed_of                        │ entails
           ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      ELEMENTS LEVEL                         │
│   Objects    │    Patterns    │    Locations                │
│ (red mug)    │   (place on)   │ (kitchen counter)          │
└─────────────────────────────────────────────────────────────┘
```

### Knowledge Types

#### Object Semantics
- **Ownership**: "my cup", "my laptop"
- **Preference**: User's favorite items
- **History**: Items with personal significance
- **Groups**: Collections of related objects

#### User Patterns
- **Routine**: Regular activity sequences
- **Preference**: Preferred arrangements or placements

## Components

### 1. UserProfileMemory
Hierarchical knowledge graph for storing personalized knowledge.

```python
from methods.MEMENTO import UserProfileMemory

memory = UserProfileMemory()
memory.add_user("user_0", "James")
memory.add_knowledge(
    "k1",
    "my_dinnerware",
    KnowledgeType.OBJECT_SEMANTICS,
    "user_0",
    alias="my dinnerware set"
)
```

### 2. KnowledgeGraphBuilder
Build and update memory from user interactions.

```python
from methods.MEMENTO import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder(memory=memory)
builder.update(instruction="Move my coffee set to the table", user_id="user_0")
```

### 3. MementoRetriever
Retrieve relevant personalized knowledge.

```python
from methods.MEMENTO import MementoRetriever

retriever = MementoRetriever(memory=memory)
results = retriever.retrieve("Arrange the items for breakfast")
```

### 4. MementoPlanner
Integration with LLM planner for personalized task execution.

```python
from methods.MEMENTO import MementoPlanner

planner = MementoPlanner(memory=memory)
goal = planner.derive_goal("Set up my morning routine")
```

## Usage

### Building Memory from Trajectories

```bash
# Using the build script directly
python methods/MEMENTO/build_memory.py \
    --data_dir data/trajectories/heuristic_results \
    --output_dir data/memento_memory \
    --user_id user_0

# Using SLURM
sbatch methods/MEMENTO/run_build_memory.slurm
```

### Integration with Planner

```python
from methods.MEMENTO import MementoRAGIntegration

# Create RAG-compatible integration
rag = MementoRAGIntegration(
    memory_path="data/memento_memory/user_profile_memory.json.gz"
)

# Retrieve personalized context
scores, indices = rag.retrieve_top_k_given_query(
    query="Move my favorite items",
    top_k=3
)

# Format for prompt
context = rag.format_examples_for_prompt(indices)
```

## Algorithms

### Knowledge Graph Update Algorithm

```
Input: User instruction I, Knowledge graph G
Output: Updated knowledge graph G'

1. EXTRACT CANDIDATE KNOWLEDGE
   K <- ExtractKnowledge(I)

2. FOR EACH KNOWLEDGE (k, t_k) in K:
   a. EXTRACT ELEMENTS
   b. SELECT SUBGRAPH BY TYPE
   c. RETRIEVE CANDIDATES via similarity search

3. LLM RESOLUTION
   Determine whether to ADD new or UPDATE existing knowledge
```

### Retrieval Procedure

```
Input: User instruction I, User profile memory graph G
Output: Retrieved natural language descriptions R

1. PARSE INSTRUCTION
2. FILTER AND SEARCH by knowledge type
3. REFORMULATE RESULTS to natural language
```

## Configuration

### Embedding Model
Default: `all-mpnet-base-v2` (Sentence Transformer)

### Retrieval Parameters
- `top_k`: Number of memories to retrieve (default: 5)
- `similarity_threshold`: Minimum similarity for retrieval (default: 0.3)

## Files

- `__init__.py`: Module exports
- `user_profile_memory.py`: Knowledge graph data structures
- `knowledge_graph_builder.py`: Memory construction and update
- `memento_retriever.py`: Retrieval procedure implementation
- `memento_planner.py`: Planner integration
- `build_memory.py`: Script for building memory from trajectories
- `run_build_memory.slurm`: SLURM job script

## Reference

```bibtex
@article{kwon2025embodied,
  title={Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization},
  author={Kwon, T. and others},
  journal={arXiv preprint arXiv:2505.16348v2},
  year={2025}
}
```

Project Website: https://connoriginal.github.io/MEMENTO
