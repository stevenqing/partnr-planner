#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import re
import shutil

from tqdm import tqdm

from enhanced_skill_extractor import EnhancedSkillExtractor


def enhance_dataset_with_enhanced_skills():
    """Enhanced dataset enhancement with improved skill extraction"""

    # Input and output paths
    input_dir = "data/rag_datasets/rerange_only_converted"
    output_dir = "data/rag_datasets/rerange_only_with_enhanced_skills"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("Enhancing rerange_only_converted dataset with ENHANCED skill extraction...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize enhanced skill extractor
    skill_extractor = EnhancedSkillExtractor(use_llm=True)
    print("✓ Enhanced skill extractor initialized")

    # Load existing metadata
    with gzip.open(
        f"{input_dir}/react_trajectories.json.gz", "rt", encoding="utf-8"
    ) as f:
        original_data = json.load(f)

    print(f"Found {len(original_data['episodes'])} episodes in original dataset")

    # Copy episode_result_log.csv unchanged
    shutil.copy2(
        f"{input_dir}/episode_result_log.csv", f"{output_dir}/episode_result_log.csv"
    )

    # Process episodes and add enhanced skills
    enhanced_episodes = []
    processed_count = 0
    skill_extraction_stats = {
        "total_processed": 0,
        "enhanced_extractions": 0,
        "fallback_extractions": 0,
        "skill_patterns_found": 0,
        "decision_points_found": 0,
        "coordination_episodes": 0,
    }

    # Create progress bar
    pbar = tqdm(
        original_data["episodes"],
        desc="Processing episodes with enhanced skills",
        unit="episode",
    )

    for episode in pbar:
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]

        try:
            # Process trace files for both agents with enhanced extraction
            agent_enhanced_skills = {}
            episode_skill_patterns = []
            episode_decision_points = []
            episode_coordination_data = {}

            for agent_id in ["0", "1"]:
                trace_file = f"trace-episode_{episode_id}_0-{agent_id}.txt"
                input_trace_path = (
                    f"{input_dir}/react_trajectories/traces/{agent_id}/{trace_file}"
                )
                output_trace_path = (
                    f"{output_dir}/react_trajectories/traces/{agent_id}/{trace_file}"
                )

                if os.path.exists(input_trace_path):
                    # Read original trace
                    with open(input_trace_path, "r", encoding="utf-8") as f:
                        trace_content = f.read()

                    # Enhanced skill extraction
                    enhanced_skill_data = skill_extractor.extract_enhanced_skills(
                        trace_content, agent_id, instruction
                    )

                    agent_enhanced_skills[agent_id] = enhanced_skill_data

                    # Accumulate episode-level data
                    episode_skill_patterns.extend(enhanced_skill_data["skill_patterns"])
                    episode_decision_points.extend(
                        enhanced_skill_data["decision_points"]
                    )
                    episode_coordination_data[agent_id] = enhanced_skill_data[
                        "coordination_requirements"
                    ]

                    # Update statistics
                    skill_extraction_stats["skill_patterns_found"] += len(
                        enhanced_skill_data["skill_patterns"]
                    )
                    skill_extraction_stats["decision_points_found"] += len(
                        enhanced_skill_data["decision_points"]
                    )
                    if enhanced_skill_data["coordination_requirements"][
                        "requires_coordination"
                    ]:
                        skill_extraction_stats["coordination_episodes"] += 1

                    # Create enhanced trace content
                    enhanced_content = create_enhanced_trace_content(
                        trace_content, enhanced_skill_data, agent_id, instruction
                    )

                    # Save enhanced trace
                    with open(output_trace_path, "w", encoding="utf-8") as f:
                        f.write(enhanced_content)

            # Create comprehensive enhanced episode info
            enhanced_episode = episode.copy()  # Preserve all original information

            # Add enhanced skill information
            enhanced_episode.update(
                {
                    "task_type": analyze_enhanced_task_type(
                        instruction, episode_skill_patterns
                    ),
                    "complexity": analyze_enhanced_complexity(
                        instruction, episode_skill_patterns, episode_decision_points
                    ),
                    "enhanced_skills": agent_enhanced_skills,
                    "skill_patterns": episode_skill_patterns,
                    "decision_points": episode_decision_points,
                    "coordination_analysis": episode_coordination_data,
                    "skill_categories": extract_enhanced_skill_categories(
                        agent_enhanced_skills
                    ),
                    "coordination_required": any(
                        data["requires_coordination"]
                        for data in episode_coordination_data.values()
                    ),
                    "episode_insights": generate_episode_insights(
                        agent_enhanced_skills,
                        episode_skill_patterns,
                        episode_decision_points,
                    ),
                    "quality_metrics": calculate_episode_quality_metrics(
                        agent_enhanced_skills
                    ),
                }
            )

            enhanced_episodes.append(enhanced_episode)
            processed_count += 1
            skill_extraction_stats["total_processed"] += 1
            skill_extraction_stats["enhanced_extractions"] += 1

            # Update progress bar with enhanced stats
            pbar.set_postfix(
                {
                    "Processed": processed_count,
                    "Episode": episode_id,
                    "Patterns": len(episode_skill_patterns),
                    "Decisions": len(episode_decision_points),
                    "Success": f"{processed_count}/{len(original_data['episodes'])}",
                }
            )

        except Exception as e:
            pbar.write(f"Error processing episode {episode_id}: {e}")
            # Keep original episode if enhanced processing fails
            enhanced_episodes.append(episode)
            processed_count += 1
            skill_extraction_stats["total_processed"] += 1
            skill_extraction_stats["fallback_extractions"] += 1

    # Create enhanced metadata with comprehensive statistics
    enhanced_metadata = original_data["metadata"].copy()  # Preserve original metadata
    enhanced_metadata.update(
        {
            "enhanced_with_skills": True,
            "enhancement_version": "v2.0_enhanced",
            "enhancement_date": "2025-09-03",
            "enhanced_by": "enhance_dataset_with_skills_v2.py",
            "skill_extractor": "EnhancedSkillExtractor",
            "skill_model": "Llama 3.1 8B + Enhanced Heuristics",
            "description": enhanced_metadata["description"]
            + " - Enhanced with multi-stage skill analysis",
            "skill_extraction_stats": skill_extraction_stats,
            "enhanced_skill_statistics": analyze_enhanced_skill_statistics(
                enhanced_episodes
            ),
            "quality_improvements": {
                "multi_stage_analysis": True,
                "decision_point_detection": True,
                "coordination_analysis": True,
                "skill_pattern_extraction": True,
                "contextual_insights": True,
            },
        }
    )

    # Create enhanced dataset
    enhanced_data = {"metadata": enhanced_metadata, "episodes": enhanced_episodes}

    # Save enhanced dataset
    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    # Generate comprehensive report
    generate_enhancement_report(output_dir, skill_extraction_stats, enhanced_metadata)

    print("\n🎉 Enhanced dataset enhancement completed!")
    print(f"Successfully processed: {processed_count} episodes")
    print(f"Enhanced extractions: {skill_extraction_stats['enhanced_extractions']}")
    print(f"Fallback extractions: {skill_extraction_stats['fallback_extractions']}")
    print(f"Total skill patterns: {skill_extraction_stats['skill_patterns_found']}")
    print(f"Total decision points: {skill_extraction_stats['decision_points_found']}")
    print(f"Coordination episodes: {skill_extraction_stats['coordination_episodes']}")
    print(f"Output directory: {output_dir}")
    print(f"Enhanced dataset file: {output_json_path}")

    return output_dir


def create_enhanced_trace_content(
    original_content: str, enhanced_skill_data: dict, agent_id: str, instruction: str
) -> str:
    """Create enhanced trace content with comprehensive skill information"""

    lines = original_content.split("\n")

    # Create enhanced content by adding comprehensive skill information at the top
    enhanced_lines = []
    enhanced_lines.append("=== ENHANCED SKILL ANALYSIS v2.0 ===")
    enhanced_lines.append(f"Agent ID: Agent_{agent_id}")
    enhanced_lines.append(f"Task Instruction: {instruction}")
    enhanced_lines.append("")

    # Primary skill description
    enhanced_lines.append("PRIMARY SKILLS:")
    enhanced_lines.append(f"  {enhanced_skill_data['enhanced_skill_description']}")
    enhanced_lines.append("")

    # Skill categories
    enhanced_lines.append("SKILL CATEGORIES:")
    for category in enhanced_skill_data["skill_categories"]:
        enhanced_lines.append(f"  - {category}")
    enhanced_lines.append("")

    # Skill patterns
    if enhanced_skill_data["skill_patterns"]:
        enhanced_lines.append("IDENTIFIED SKILL PATTERNS:")
        for i, pattern in enumerate(enhanced_skill_data["skill_patterns"], 1):
            enhanced_lines.append(
                f"  {i}. {pattern['skill_name']} ({pattern['skill_type']})"
            )
            enhanced_lines.append(f"     Description: {pattern['description']}")
            if pattern["success_indicators"]:
                enhanced_lines.append(
                    f"     Success Indicators: {', '.join(pattern['success_indicators'])}"
                )
        enhanced_lines.append("")

    # Decision points
    if enhanced_skill_data["decision_points"]:
        enhanced_lines.append("CRITICAL DECISION POINTS:")
        for i, decision in enumerate(enhanced_skill_data["decision_points"], 1):
            enhanced_lines.append(
                f"  {i}. Step {decision['step_number']}: {decision['decision_type']}"
            )
            enhanced_lines.append(f"     Context: {decision['context']}")
            enhanced_lines.append(f"     Reasoning: {decision['reasoning']}")
        enhanced_lines.append("")

    # Coordination analysis
    coord_req = enhanced_skill_data["coordination_requirements"]
    if (
        coord_req["requires_coordination"]
        or coord_req["coordination_actions_count"] > 0
    ):
        enhanced_lines.append("COORDINATION ANALYSIS:")
        enhanced_lines.append(
            f"  Requires Coordination: {coord_req['requires_coordination']}"
        )
        enhanced_lines.append(
            f"  Coordination Actions: {coord_req['coordination_actions_count']}"
        )
        enhanced_lines.append(f"  Wait Actions: {coord_req['wait_actions_count']}")
        enhanced_lines.append(
            f"  Effectiveness: {coord_req['coordination_effectiveness']}"
        )
        enhanced_lines.append("")

    # Efficiency metrics
    efficiency = enhanced_skill_data["action_efficiency"]
    enhanced_lines.append("EFFICIENCY METRICS:")
    enhanced_lines.append(f"  Overall Score: {efficiency['efficiency_score']:.3f}")
    enhanced_lines.append(
        f"  Success Rate: {efficiency['metrics']['success_rate']:.3f}"
    )
    enhanced_lines.append(
        f"  Action Diversity: {efficiency['metrics']['action_diversity']:.3f}"
    )
    enhanced_lines.append(
        f"  Steps: {efficiency['metrics']['successful_steps']}/{efficiency['metrics']['total_steps']}"
    )
    enhanced_lines.append("")

    # Learning insights
    if enhanced_skill_data["learning_insights"]:
        enhanced_lines.append("LEARNING INSIGHTS:")
        for insight in enhanced_skill_data["learning_insights"]:
            enhanced_lines.append(f"  - {insight}")
        enhanced_lines.append("")

    enhanced_lines.append("=== ORIGINAL TRACE ===")
    enhanced_lines.append("")
    enhanced_lines.extend(lines)  # Preserve all original content

    return "\n".join(enhanced_lines)


def analyze_enhanced_task_type(instruction: str, skill_patterns: list) -> str:
    """Analyze task type with enhanced pattern recognition"""
    instruction_lower = instruction.lower()

    # Check skill patterns for more specific categorization
    pattern_types = [pattern["skill_type"] for pattern in skill_patterns]

    # Enhanced room-based categorization
    if any(word in instruction_lower for word in ["bedroom", "bed"]):
        return "Bedroom Organization"
    elif any(word in instruction_lower for word in ["kitchen", "cook", "food"]):
        return "Kitchen Organization"
    elif any(word in instruction_lower for word in ["dining", "table", "dinner"]):
        return "Dining Room Setup"
    elif any(word in instruction_lower for word in ["living room", "couch", "sofa"]):
        return "Living Room Organization"
    elif any(word in instruction_lower for word in ["bathroom", "toilet", "shower"]):
        return "Bathroom Organization"
    elif any(word in instruction_lower for word in ["hallway", "corridor"]):
        return "Hallway Organization"

    # Pattern-based categorization
    elif "coordination" in pattern_types:
        return "Multi-Agent Coordination Task"
    elif "manipulation" in pattern_types and "navigation" in pattern_types:
        return "Complex Multi-Object Movement"
    elif instruction_lower.count("move") > 2 or instruction_lower.count("and") > 2:
        return "Complex Multi-Step Task"
    elif "move" in instruction_lower and (
        "from" in instruction_lower or "to" in instruction_lower
    ):
        return "Object Relocation Task"
    elif any(word in instruction_lower for word in ["clean", "organize", "tidy"]):
        return "General Cleaning/Organization"
    else:
        return "General Task"


def analyze_enhanced_complexity(
    instruction: str, skill_patterns: list, decision_points: list
) -> str:
    """Analyze task complexity with enhanced metrics"""
    instruction_lower = instruction.lower()

    # Basic instruction analysis
    object_mentions = len(
        re.findall(r"\b(?:move|place|put|take|get)\s+\w+", instruction_lower)
    )
    conjunction_count = instruction_lower.count("and") + instruction_lower.count("or")
    word_count = len(instruction.split())

    # Enhanced pattern analysis
    pattern_diversity = len(set(pattern["skill_type"] for pattern in skill_patterns))
    decision_complexity = len(
        [
            dp
            for dp in decision_points
            if dp["decision_type"] in ["recovery", "coordination"]
        ]
    )

    # Calculate complexity score
    complexity_score = 0
    complexity_score += min(object_mentions * 2, 10)  # Object complexity (max 10)
    complexity_score += min(conjunction_count * 3, 9)  # Conjunction complexity (max 9)
    complexity_score += min(word_count / 4, 8)  # Instruction length (max 8)
    complexity_score += min(pattern_diversity * 3, 12)  # Pattern diversity (max 12)
    complexity_score += min(decision_complexity * 4, 16)  # Decision complexity (max 16)

    # Categorize based on score
    if complexity_score >= 35:
        return "Very High"
    elif complexity_score >= 25:
        return "High"
    elif complexity_score >= 15:
        return "Medium"
    elif complexity_score >= 8:
        return "Low"
    else:
        return "Very Low"


def extract_enhanced_skill_categories(agent_enhanced_skills: dict) -> list:
    """Extract enhanced skill categories from all agents"""
    all_categories = set()

    for _agent_id, skill_data in agent_enhanced_skills.items():
        if "skill_categories" in skill_data:
            all_categories.update(skill_data["skill_categories"])

    return list(all_categories)


def generate_episode_insights(
    agent_skills: dict, skill_patterns: list, decision_points: list
) -> list:
    """Generate comprehensive episode-level insights"""
    insights = []

    # Multi-agent insights
    if len(agent_skills) > 1:
        agent_efficiency = {}
        for agent_id, skills in agent_skills.items():
            efficiency = skills["action_efficiency"]["efficiency_score"]
            agent_efficiency[agent_id] = efficiency

        best_agent = max(agent_efficiency.keys(), key=lambda k: agent_efficiency[k])
        insights.append(
            f"Agent {best_agent} demonstrated highest efficiency ({agent_efficiency[best_agent]:.3f})"
        )

    # Pattern insights
    pattern_types = [p["skill_type"] for p in skill_patterns]
    if len(set(pattern_types)) > 2:
        insights.append(
            f"Demonstrates versatile skills across {len(set(pattern_types))} different skill types"
        )

    # Decision-making insights
    recovery_decisions = [
        dp for dp in decision_points if dp["decision_type"] == "recovery"
    ]
    if recovery_decisions:
        insights.append(
            f"Shows adaptive behavior with {len(recovery_decisions)} successful recovery strategies"
        )

    coordination_decisions = [
        dp for dp in decision_points if dp["decision_type"] == "coordination"
    ]
    if coordination_decisions:
        insights.append(
            f"Demonstrates multi-agent coordination with {len(coordination_decisions)} coordination points"
        )

    return insights


def calculate_episode_quality_metrics(agent_skills: dict) -> dict:
    """Calculate comprehensive quality metrics for the episode"""
    if not agent_skills:
        return {"overall_quality": 0.0, "metrics": {}}

    # Aggregate efficiency scores
    efficiency_scores = [
        skills["action_efficiency"]["efficiency_score"]
        for skills in agent_skills.values()
        if "action_efficiency" in skills
    ]

    # Aggregate success rates
    success_rates = [
        skills["action_efficiency"]["metrics"]["success_rate"]
        for skills in agent_skills.values()
        if "action_efficiency" in skills and "metrics" in skills["action_efficiency"]
    ]

    # Count total patterns and insights
    total_patterns = sum(
        len(skills.get("skill_patterns", [])) for skills in agent_skills.values()
    )

    total_insights = sum(
        len(skills.get("learning_insights", [])) for skills in agent_skills.values()
    )

    # Calculate overall quality
    avg_efficiency = (
        sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
    )
    avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
    pattern_bonus = min(total_patterns * 0.05, 0.2)  # Max 0.2 bonus
    insight_bonus = min(total_insights * 0.02, 0.1)  # Max 0.1 bonus

    overall_quality = (
        avg_efficiency * 0.6 + avg_success_rate * 0.4 + pattern_bonus + insight_bonus
    )

    return {
        "overall_quality": overall_quality,
        "metrics": {
            "average_efficiency": avg_efficiency,
            "average_success_rate": avg_success_rate,
            "total_skill_patterns": total_patterns,
            "total_insights": total_insights,
            "pattern_bonus": pattern_bonus,
            "insight_bonus": insight_bonus,
        },
    }


def analyze_enhanced_skill_statistics(episodes: list) -> dict:
    """Analyze comprehensive skill statistics across the enhanced dataset"""
    task_type_counts = {}
    complexity_counts = {}
    skill_category_counts = {}
    pattern_type_counts = {}
    decision_type_counts = {}
    quality_distribution = {"very_high": 0, "high": 0, "medium": 0, "low": 0}

    for episode in episodes:
        # Task types
        task_type = episode.get("task_type", "Unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # Complexity
        complexity = episode.get("complexity", "Unknown")
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Skill categories
        skill_categories = episode.get("skill_categories", [])
        for category in skill_categories:
            skill_category_counts[category] = skill_category_counts.get(category, 0) + 1

        # Skill patterns
        skill_patterns = episode.get("skill_patterns", [])
        for pattern in skill_patterns:
            pattern_type = pattern.get("skill_type", "unknown")
            pattern_type_counts[pattern_type] = (
                pattern_type_counts.get(pattern_type, 0) + 1
            )

        # Decision types
        decision_points = episode.get("decision_points", [])
        for decision in decision_points:
            decision_type = decision.get("decision_type", "unknown")
            decision_type_counts[decision_type] = (
                decision_type_counts.get(decision_type, 0) + 1
            )

        # Quality distribution
        quality_metrics = episode.get("quality_metrics", {})
        quality_score = quality_metrics.get("overall_quality", 0)
        if quality_score >= 0.8:
            quality_distribution["very_high"] += 1
        elif quality_score >= 0.6:
            quality_distribution["high"] += 1
        elif quality_score >= 0.4:
            quality_distribution["medium"] += 1
        else:
            quality_distribution["low"] += 1

    return {
        "task_type_distribution": task_type_counts,
        "complexity_distribution": complexity_counts,
        "skill_category_distribution": skill_category_counts,
        "skill_pattern_distribution": pattern_type_counts,
        "decision_type_distribution": decision_type_counts,
        "quality_distribution": quality_distribution,
        "total_episodes_enhanced": len(
            [e for e in episodes if e.get("enhanced_skills")]
        ),
        "average_patterns_per_episode": sum(
            len(e.get("skill_patterns", [])) for e in episodes
        )
        / len(episodes)
        if episodes
        else 0,
        "average_decisions_per_episode": sum(
            len(e.get("decision_points", [])) for e in episodes
        )
        / len(episodes)
        if episodes
        else 0,
    }


def generate_enhancement_report(output_dir: str, stats: dict, metadata: dict):
    """Generate a comprehensive enhancement report"""

    report_content = f"""# Enhanced Skill Extraction Report v2.0

## 🎯 Enhancement Summary

The dataset has been successfully enhanced using the new **EnhancedSkillExtractor** with multi-stage analysis.

### Processing Statistics
- **Total Episodes Processed**: {stats['total_processed']}
- **Enhanced Extractions**: {stats['enhanced_extractions']}
- **Fallback Extractions**: {stats['fallback_extractions']}
- **Success Rate**: {(stats['enhanced_extractions']/stats['total_processed']*100):.1f}%

### Skill Analysis Results
- **Total Skill Patterns Identified**: {stats['skill_patterns_found']}
- **Total Decision Points Detected**: {stats['decision_points_found']}
- **Episodes Requiring Coordination**: {stats['coordination_episodes']}
- **Average Patterns per Episode**: {stats['skill_patterns_found']/stats['total_processed']:.1f}
- **Average Decisions per Episode**: {stats['decision_points_found']/stats['total_processed']:.1f}

## 🚀 Enhanced Features

### 1. Multi-Stage Skill Analysis
- **Stage 1**: Parse trace into structured action steps
- **Stage 2**: Identify critical decision points
- **Stage 3**: Extract skill patterns (navigation, manipulation, coordination, planning)
- **Stage 4**: Generate contextual skill descriptions
- **Stage 5**: Analyze coordination requirements

### 2. Decision Point Detection
- Exploration decisions when previous attempts fail
- Recovery strategies after action failures
- Multi-agent coordination timing
- Alternative strategy selection

### 3. Skill Pattern Recognition
- Systematic exploration patterns
- Object manipulation sequences
- Recovery and adaptation strategies
- Coordination protocols

### 4. Comprehensive Metrics
- Action efficiency scoring
- Success rate analysis
- Coordination effectiveness
- Quality assessments

## 📊 Quality Improvements

The enhanced extraction provides:
- **Detailed skill descriptions** with specific actions and locations
- **Decision reasoning** explaining why agents chose specific actions
- **Pattern recognition** identifying reusable strategies
- **Coordination analysis** for multi-agent scenarios
- **Learning insights** for training and evaluation

## 🔧 Usage

The enhanced dataset is ready for use with RAG systems:

```python
# Load enhanced dataset
with gzip.open('data/rag_datasets/rerange_only_with_enhanced_skills/react_trajectories.json.gz', 'rt') as f:
    enhanced_data = json.load(f)

# Access enhanced skill information
for episode in enhanced_data['episodes']:
    enhanced_skills = episode['enhanced_skills']  # Per-agent enhanced skills
    skill_patterns = episode['skill_patterns']    # Identified patterns
    decision_points = episode['decision_points']  # Critical decisions
    insights = episode['episode_insights']        # Learning insights
```

## 🎉 Next Steps

1. **Test RAG Integration**: Use enhanced skills in retrieval and prompting
2. **Evaluate Quality**: Compare performance against baseline skill extraction
3. **Analyze Patterns**: Study common skill patterns for curriculum development
4. **Expand Dataset**: Apply enhanced extraction to larger trajectory collections

---
Generated by: enhance_dataset_with_skills_v2.py
Date: 2025-09-03
Version: EnhancedSkillExtractor v2.0
"""

    # Save report
    report_path = os.path.join(output_dir, "ENHANCED_SKILL_EXTRACTION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"📄 Enhancement report saved: {report_path}")


if __name__ == "__main__":
    output_dir = enhance_dataset_with_enhanced_skills()
    print(
        f"\n✅ Enhanced skill extraction completed! Dataset available at: {output_dir}"
    )
