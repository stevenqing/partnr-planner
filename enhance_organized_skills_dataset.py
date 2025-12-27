#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import gzip
import json
import os

from tqdm import tqdm

from enhanced_skill_extractor import EnhancedSkillExtractor


def enhance_organized_skills_dataset():
    """Apply enhanced skill extraction to the organized skills dataset"""

    input_dir = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_organized_by_skills"
    output_dir = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_organized_by_skills_enhanced"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/by_complexity", exist_ok=True)
    os.makedirs(f"{output_dir}/by_task_type", exist_ok=True)

    print("🚀 Enhancing organized skills dataset with advanced skill extraction")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize enhanced skill extractor
    extractor = EnhancedSkillExtractor(use_llm=True)
    print("✓ Enhanced skill extractor initialized")

    # Find all skill-based dataset files
    skill_files = glob.glob(f"{input_dir}/skill_*.json.gz")
    complexity_files = glob.glob(f"{input_dir}/by_complexity/*.json.gz")
    task_type_files = glob.glob(f"{input_dir}/by_task_type/*.json.gz")

    all_files = skill_files + complexity_files + task_type_files
    print(f"Found {len(all_files)} organized dataset files to enhance")

    # Load the original trajectory data source for trace content
    original_traces_dir = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_with_skills/react_trajectories/traces"

    enhancement_stats = {
        "total_files_processed": 0,
        "total_episodes_enhanced": 0,
        "enhanced_extractions": 0,
        "fallback_extractions": 0,
        "skill_patterns_found": 0,
        "decision_points_found": 0,
        "coordination_episodes": 0,
    }

    # Process each organized dataset file
    for file_path in tqdm(all_files, desc="Enhancing organized datasets", unit="file"):
        try:
            # Determine output path
            relative_path = os.path.relpath(file_path, input_dir)
            output_file_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Load the organized dataset
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                dataset = json.load(f)

            print(f"\n📊 Processing: {os.path.basename(file_path)}")
            print(f"   Episodes: {len(dataset['episodes'])}")
            print(
                f"   Skill Category: {dataset['metadata'].get('skill_category', 'N/A')}"
            )

            # Enhance episodes in this dataset
            enhanced_episodes = []
            file_stats = {
                "episodes_processed": 0,
                "enhanced_successfully": 0,
                "patterns_found": 0,
                "decisions_found": 0,
            }

            episode_pbar = tqdm(
                dataset["episodes"],
                desc=f"  Enhancing {os.path.basename(file_path)}",
                leave=False,
                unit="episode",
            )

            for episode in episode_pbar:
                episode_id = episode["episode_id"]
                episode["instruction"]

                try:
                    # Apply enhanced skill extraction
                    enhanced_episode_data = enhance_episode_with_advanced_skills(
                        episode, extractor, original_traces_dir
                    )

                    if enhanced_episode_data:
                        enhanced_episodes.append(enhanced_episode_data)
                        file_stats["enhanced_successfully"] += 1
                        file_stats["patterns_found"] += len(
                            enhanced_episode_data.get("skill_patterns", [])
                        )
                        file_stats["decisions_found"] += len(
                            enhanced_episode_data.get("decision_points", [])
                        )

                        # Check for coordination
                        coordination_data = enhanced_episode_data.get(
                            "coordination_analysis", {}
                        )
                        if any(
                            agent_coord.get("requires_coordination", False)
                            for agent_coord in coordination_data.values()
                        ):
                            enhancement_stats["coordination_episodes"] += 1
                    else:
                        # Keep original episode if enhancement fails
                        enhanced_episodes.append(episode)
                        enhancement_stats["fallback_extractions"] += 1

                    file_stats["episodes_processed"] += 1

                    # Update progress
                    episode_pbar.set_postfix(
                        {
                            "Enhanced": file_stats["enhanced_successfully"],
                            "Patterns": file_stats["patterns_found"],
                            "Decisions": file_stats["decisions_found"],
                        }
                    )

                except Exception as e:
                    tqdm.write(f"    ⚠️ Error enhancing episode {episode_id}: {e}")
                    enhanced_episodes.append(episode)  # Keep original
                    enhancement_stats["fallback_extractions"] += 1
                    file_stats["episodes_processed"] += 1

            # Update enhanced dataset metadata
            enhanced_metadata = dataset["metadata"].copy()
            enhanced_metadata.update(
                {
                    "enhanced_with_advanced_skills": True,
                    "enhancement_version": "v2.0_enhanced",
                    "enhancement_date": "2025-09-03",
                    "enhanced_by": "enhance_organized_skills_dataset.py",
                    "original_episodes": len(dataset["episodes"]),
                    "enhanced_episodes": file_stats["enhanced_successfully"],
                    "enhancement_stats": file_stats,
                    "skill_extractor": "EnhancedSkillExtractor v2.0",
                    "description": enhanced_metadata.get("description", "")
                    + " - Enhanced with multi-stage skill analysis",
                }
            )

            # Create enhanced dataset
            enhanced_dataset = {
                "metadata": enhanced_metadata,
                "episodes": enhanced_episodes,
            }

            # Save enhanced dataset
            with gzip.open(output_file_path, "wt", encoding="utf-8") as f:
                json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)

            # Update overall stats
            enhancement_stats["total_files_processed"] += 1
            enhancement_stats["total_episodes_enhanced"] += file_stats[
                "enhanced_successfully"
            ]
            enhancement_stats["enhanced_extractions"] += file_stats[
                "enhanced_successfully"
            ]
            enhancement_stats["skill_patterns_found"] += file_stats["patterns_found"]
            enhancement_stats["decision_points_found"] += file_stats["decisions_found"]

            print(
                f"   ✓ Enhanced: {file_stats['enhanced_successfully']}/{file_stats['episodes_processed']} episodes"
            )
            print(f"   ✓ Saved to: {output_file_path}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    # Copy and enhance organization summary
    enhance_organization_summary(input_dir, output_dir, enhancement_stats)

    # Generate comprehensive report
    generate_organized_enhancement_report(output_dir, enhancement_stats)

    print("\n🎉 Enhanced organized skills dataset completed!")
    print(f"Files processed: {enhancement_stats['total_files_processed']}")
    print(f"Episodes enhanced: {enhancement_stats['total_episodes_enhanced']}")
    print(f"Skill patterns found: {enhancement_stats['skill_patterns_found']}")
    print(f"Decision points found: {enhancement_stats['decision_points_found']}")
    print(f"Coordination episodes: {enhancement_stats['coordination_episodes']}")
    print(f"Output directory: {output_dir}")

    return output_dir


def enhance_episode_with_advanced_skills(episode, extractor, traces_dir):
    """Apply enhanced skill extraction to a single episode"""

    episode_id = episode["episode_id"]
    instruction = episode["instruction"]

    # Enhanced skill data for both agents
    agent_enhanced_skills = {}
    episode_skill_patterns = []
    episode_decision_points = []
    episode_coordination_data = {}

    for agent_id in ["0", "1"]:
        trace_file = f"trace-episode_{episode_id}_0-{agent_id}.txt"
        trace_path = f"{traces_dir}/{agent_id}/{trace_file}"

        if os.path.exists(trace_path):
            try:
                # Read trace content
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace_content = f.read()

                # Apply enhanced extraction
                enhanced_skill_data = extractor.extract_enhanced_skills(
                    trace_content, agent_id, instruction
                )

                agent_enhanced_skills[agent_id] = enhanced_skill_data

                # Accumulate episode-level data
                episode_skill_patterns.extend(enhanced_skill_data["skill_patterns"])
                episode_decision_points.extend(enhanced_skill_data["decision_points"])
                episode_coordination_data[agent_id] = enhanced_skill_data[
                    "coordination_requirements"
                ]

            except Exception as e:
                print(f"      ⚠️ Error processing agent {agent_id} trace: {e}")
                continue

    if not agent_enhanced_skills:
        return None  # No successful extractions

    # Create comprehensive enhanced episode
    enhanced_episode = episode.copy()  # Preserve all original data

    # Add enhanced skill information
    enhanced_episode.update(
        {
            "enhanced_skills": agent_enhanced_skills,
            "skill_patterns": episode_skill_patterns,
            "decision_points": episode_decision_points,
            "coordination_analysis": episode_coordination_data,
            "enhanced_skill_categories": extract_enhanced_categories_from_patterns(
                episode_skill_patterns
            ),
            "episode_quality_metrics": calculate_episode_quality_from_agents(
                agent_enhanced_skills
            ),
            "episode_insights": generate_episode_insights_from_patterns(
                agent_enhanced_skills, episode_skill_patterns, episode_decision_points
            ),
            "coordination_required": any(
                coord_data.get("requires_coordination", False)
                for coord_data in episode_coordination_data.values()
            ),
            "enhancement_metadata": {
                "enhanced_with": "EnhancedSkillExtractor v2.0",
                "enhancement_date": "2025-09-03",
                "agents_processed": list(agent_enhanced_skills.keys()),
                "total_patterns": len(episode_skill_patterns),
                "total_decisions": len(episode_decision_points),
            },
        }
    )

    return enhanced_episode


def extract_enhanced_categories_from_patterns(skill_patterns):
    """Extract enhanced skill categories from skill patterns"""
    categories = set()

    for pattern in skill_patterns:
        skill_type = pattern.get("skill_type", "unknown")

        if skill_type == "navigation":
            categories.add("Advanced Navigation")
        elif skill_type == "manipulation":
            categories.add("Object Manipulation Mastery")
        elif skill_type == "coordination":
            categories.add("Multi-Agent Coordination")
        elif skill_type == "planning":
            categories.add("Adaptive Planning")

        # Add specific skill categories
        skill_name = pattern.get("skill_name", "")
        if "systematic_exploration" in skill_name:
            categories.add("Systematic Exploration")
        elif "pick_and_place" in skill_name:
            categories.add("Pick-and-Place Expertise")
        elif "failure_recovery" in skill_name:
            categories.add("Failure Recovery")
        elif "multi_agent_coordination" in skill_name:
            categories.add("Coordination Protocols")

    return list(categories)


def calculate_episode_quality_from_agents(agent_skills):
    """Calculate quality metrics from agent skill data"""
    if not agent_skills:
        return {"overall_quality": 0.0, "metrics": {}}

    # Aggregate metrics across agents
    total_efficiency = 0
    total_success_rate = 0
    total_patterns = 0
    total_insights = 0
    agent_count = len(agent_skills)

    for _agent_id, skills in agent_skills.items():
        efficiency_data = skills.get("action_efficiency", {})
        total_efficiency += efficiency_data.get("efficiency_score", 0)

        metrics = efficiency_data.get("metrics", {})
        total_success_rate += metrics.get("success_rate", 0)

        total_patterns += len(skills.get("skill_patterns", []))
        total_insights += len(skills.get("learning_insights", []))

    # Calculate averages
    avg_efficiency = total_efficiency / agent_count if agent_count > 0 else 0
    avg_success_rate = total_success_rate / agent_count if agent_count > 0 else 0

    # Quality bonuses
    pattern_bonus = min(total_patterns * 0.05, 0.2)
    insight_bonus = min(total_insights * 0.02, 0.1)

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
            "agent_count": agent_count,
            "pattern_bonus": pattern_bonus,
            "insight_bonus": insight_bonus,
        },
    }


def generate_episode_insights_from_patterns(
    agent_skills, skill_patterns, decision_points
):
    """Generate insights from patterns and decisions"""
    insights = []

    # Pattern-based insights
    pattern_types = set(p.get("skill_type", "unknown") for p in skill_patterns)
    if len(pattern_types) > 2:
        insights.append(
            f"Demonstrates versatile skills across {len(pattern_types)} skill domains"
        )

    # Decision-making insights
    recovery_decisions = [
        dp for dp in decision_points if dp.get("decision_type") == "recovery"
    ]
    if recovery_decisions:
        insights.append(
            f"Shows resilience with {len(recovery_decisions)} successful recovery strategies"
        )

    coordination_decisions = [
        dp for dp in decision_points if dp.get("decision_type") == "coordination"
    ]
    if coordination_decisions:
        insights.append(
            f"Demonstrates coordination with {len(coordination_decisions)} multi-agent decisions"
        )

    # Multi-agent insights
    if len(agent_skills) > 1:
        efficiencies = {
            agent_id: skills.get("action_efficiency", {}).get("efficiency_score", 0)
            for agent_id, skills in agent_skills.items()
        }
        best_agent = max(efficiencies.keys(), key=lambda k: efficiencies[k])
        insights.append(
            f"Agent {best_agent} shows highest efficiency ({efficiencies[best_agent]:.3f})"
        )

    # Skill-specific insights
    navigation_patterns = [
        p for p in skill_patterns if p.get("skill_type") == "navigation"
    ]
    manipulation_patterns = [
        p for p in skill_patterns if p.get("skill_type") == "manipulation"
    ]

    if navigation_patterns and manipulation_patterns:
        insights.append("Effectively combines navigation and manipulation skills")

    return insights


def enhance_organization_summary(input_dir, output_dir, enhancement_stats):
    """Copy and enhance the organization summary"""

    input_summary_path = f"{input_dir}/organization_summary.json"
    output_summary_path = f"{output_dir}/organization_summary.json"

    try:
        with open(input_summary_path, "r") as f:
            summary = json.load(f)

        # Add enhancement information
        summary["enhancement_info"] = {
            "enhanced_with": "EnhancedSkillExtractor v2.0",
            "enhancement_date": "2025-09-03",
            "enhancement_stats": enhancement_stats,
            "description": "Original organized dataset enhanced with multi-stage skill analysis",
        }

        with open(output_summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("✓ Enhanced organization summary saved")

    except Exception as e:
        print(f"⚠️ Warning: Could not enhance organization summary: {e}")


def generate_organized_enhancement_report(output_dir, stats):
    """Generate comprehensive enhancement report"""

    report_content = f"""# Enhanced Organized Skills Dataset Report

## 🎯 Enhancement Summary

The organized skills dataset has been successfully enhanced with advanced multi-stage skill extraction.

### Enhancement Statistics
- **Files Processed**: {stats['total_files_processed']}
- **Episodes Enhanced**: {stats['total_episodes_enhanced']}
- **Enhanced Extractions**: {stats['enhanced_extractions']}
- **Fallback Extractions**: {stats['fallback_extractions']}
- **Success Rate**: {(stats['enhanced_extractions']/(stats['enhanced_extractions']+stats['fallback_extractions'])*100):.1f}%

### Advanced Analysis Results
- **Skill Patterns Identified**: {stats['skill_patterns_found']}
- **Decision Points Detected**: {stats['decision_points_found']}
- **Coordination Episodes**: {stats['coordination_episodes']}
- **Average Patterns per Episode**: {stats['skill_patterns_found']/max(stats['total_episodes_enhanced'],1):.1f}
- **Average Decisions per Episode**: {stats['decision_points_found']/max(stats['total_episodes_enhanced'],1):.1f}

## 🚀 Enhanced Features Applied

### 1. Multi-Stage Skill Analysis
Each episode now includes:
- **Structured Action Steps**: Parsed and analyzed action sequences
- **Decision Point Detection**: Critical decision moments identified
- **Skill Pattern Recognition**: Recurring successful strategies
- **Contextual Skill Descriptions**: Detailed, actionable skill summaries

### 2. Advanced Categorization
- **Enhanced Skill Categories**: Beyond basic categories to specific expertise
- **Quality Metrics**: Comprehensive efficiency and success scoring
- **Coordination Analysis**: Multi-agent interaction patterns
- **Learning Insights**: Actionable takeaways for each episode

### 3. Organized Dataset Benefits
- **Skill-Specific Enhancement**: Each organized category gets targeted analysis
- **Preserved Organization**: Original skill-based structure maintained
- **Cross-Reference Capabilities**: Episodes can be found by multiple criteria
- **Quality Filtering**: Easy identification of high-quality examples

## 📊 Dataset Organization

The enhanced dataset maintains the original organization:

### By Skill Category
- `skill_navigation.json.gz` - Enhanced navigation expertise
- `skill_object_manipulation.json.gz` - Advanced manipulation patterns
- `skill_object_placement.json.gz` - Placement strategy analysis
- `skill_container_management.json.gz` - Container interaction skills
- `skill_multi_agent_coordination.json.gz` - Coordination protocols
- `skill_room_specific_organization.json.gz` - Room-based strategies
- `skill_task_planning.json.gz` - Planning and execution patterns

### By Task Type & Complexity
- `by_task_type/` - Episodes organized by task category
- `by_complexity/` - Episodes organized by complexity level

## 🔧 Usage Examples

### Access Enhanced Skill Data
```python
import gzip
import json

# Load enhanced navigation skills dataset
with gzip.open('skill_navigation.json.gz', 'rt') as f:
    nav_data = json.load(f)

for episode in nav_data['episodes']:
    # Access enhanced skill information
    enhanced_skills = episode['enhanced_skills']
    skill_patterns = episode['skill_patterns']
    decision_points = episode['decision_points']
    quality_metrics = episode['episode_quality_metrics']
```

### Filter by Quality
```python
# Get high-quality episodes
high_quality_episodes = [
    ep for ep in episodes
    if ep['episode_quality_metrics']['overall_quality'] > 0.8
]
```

### Extract Coordination Examples
```python
# Find multi-agent coordination examples
coordination_episodes = [
    ep for ep in episodes
    if ep['coordination_required'] and len(ep['decision_points']) > 0
]
```

## 🎉 Benefits for RAG Systems

1. **Better Retrieval**: Enhanced skill descriptions improve similarity matching
2. **Richer Context**: Decision points and patterns provide deeper understanding
3. **Quality Filtering**: Quality metrics enable selection of best examples
4. **Targeted Learning**: Organized structure allows skill-specific training
5. **Coordination Insights**: Multi-agent examples for collaborative tasks

## 📈 Next Steps

1. **RAG Integration**: Test enhanced skills in retrieval and prompting
2. **Quality Analysis**: Compare performance against baseline extractions
3. **Skill Curriculum**: Design learning progressions using skill patterns
4. **Failure Analysis**: Study decision points to understand failure modes

---
Generated by: enhance_organized_skills_dataset.py
Date: 2025-09-03
Version: EnhancedSkillExtractor v2.0 + Organized Dataset Enhancement
"""

    report_path = f"{output_dir}/ENHANCED_ORGANIZED_DATASET_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"📄 Enhancement report saved: {report_path}")


if __name__ == "__main__":
    output_dir = enhance_organized_skills_dataset()
    print(f"\n✅ Enhanced organized dataset ready at: {output_dir}")
