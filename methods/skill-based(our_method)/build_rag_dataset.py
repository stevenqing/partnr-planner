#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Dataset Builder from Heuristic Trajectories

This script builds a RAG dataset from heuristic trajectory results,
integrating the hierarchical skill memory and ToM analysis.

The output dataset follows the format expected by habitat_llm/planner/rag.py

Reference: /doc/our_method
"""

import csv
import gzip
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .build_hierarchical_skill_memory import HierarchicalSkillMemory, build_memory_from_heuristic_results
from .theory_of_mind import TheoryOfMindReasoner, create_tom_reasoner


class RAGDatasetBuilder:
    """
    Build RAG dataset from heuristic trajectory results

    Combines:
    - Hierarchical skill memory (L_ind, L_coop)
    - Theory of Mind analysis
    - Episodic trajectory data
    """

    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        filter_successful: bool = True
    ):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.filter_successful = filter_successful

        # Initialize components
        self.memory = HierarchicalSkillMemory()
        self.tom_reasoner = create_tom_reasoner()

        # Dataset storage
        self.episodes = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "source": results_dir,
            "filter_successful": filter_successful,
            "total_episodes": 0,
            "cooperation_statistics": {}
        }

    def load_episode_results(self) -> Dict[str, Dict]:
        """Load episode results from CSV"""
        episode_log_path = os.path.join(self.results_dir, "episode_result_log.csv")

        if not os.path.exists(episode_log_path):
            print(f"Error: Episode log not found at {episode_log_path}")
            return {}

        episodes = {}
        with open(episode_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = row.get("episode_id", "").strip()
                success = float(row.get("task_state_success", 0))

                if self.filter_successful and success < 1.0:
                    continue

                episodes[episode_id] = {
                    "instruction": row.get("instruction", ""),
                    "success": success,
                    "task_percent_complete": float(row.get("task_percent_complete", 0)),
                    "runtime": float(row.get("runtime", 0)),
                    "sim_step_count": float(row.get("sim_step_count", 0))
                }

        print(f"Loaded {len(episodes)} episodes")
        return episodes

    def find_traces_directory(self) -> Optional[str]:
        """Find the traces directory in results"""
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path) and item.endswith(".gz"):
                traces_dir = os.path.join(item_path, "traces")
                if os.path.exists(traces_dir):
                    return traces_dir
        return None

    def process_episode(
        self,
        episode_id: str,
        episode_info: Dict,
        trace_paths: Dict[int, str]
    ) -> Optional[Dict]:
        """
        Process a single episode into RAG format

        Args:
            episode_id: Episode identifier
            episode_info: Episode metadata
            trace_paths: Dict mapping agent_id to trace file path

        Returns:
            Processed episode data or None if processing fails
        """
        if not trace_paths:
            return None

        # Parse trace file (use agent 0's trace as it contains both agents' actions)
        trace_path = trace_paths.get(0) or list(trace_paths.values())[0]

        try:
            with open(trace_path, 'r', encoding='utf-8') as f:
                trace_content = f.read()
        except Exception as e:
            print(f"Error reading trace {trace_path}: {e}")
            return None

        # Parse the trace
        task, env_state, actions = self.memory.parse_trace_file(trace_path)

        # Extract individual skills
        agent_0_skills = self.memory.extract_individual_skills(actions, env_state, 0)
        agent_1_skills = self.memory.extract_individual_skills(actions, env_state, 1)

        # Extract cooperation skills
        coop_skills = self.memory.extract_cooperation_skills(actions, env_state, task)

        # Perform ToM analysis
        tom_analysis = self.tom_reasoner.extract_tom_from_trace(trace_content, agent_id=0)

        # Build cooperation skills metadata
        cooperation_data = self._analyze_cooperation(actions, coop_skills, tom_analysis)

        # Build episode data
        episode_data = {
            "episode_id": episode_id,
            "instruction": task or episode_info.get("instruction", ""),
            "success_rate": episode_info.get("success", 1.0),
            "task_percent_complete": episode_info.get("task_percent_complete", 1.0),
            "format": "react",

            # Skill information
            "individual_skills": {
                "agent_0": [self._skill_to_dict(s) for s in agent_0_skills],
                "agent_1": [self._skill_to_dict(s) for s in agent_1_skills]
            },

            # Cooperation skills
            "cooperation_skills": cooperation_data,

            # ToM analysis
            "tom_analysis": {
                "has_tom_reasoning": len(tom_analysis) > 0,
                "tom_steps": len(tom_analysis),
                "cooperation_patterns": list(set(t.cooperation_pattern for t in tom_analysis)),
                "avg_confidence": sum(t.confidence for t in tom_analysis) / len(tom_analysis) if tom_analysis else 0
            },

            # Trace reference
            "trace_files": {str(k): os.path.basename(v) for k, v in trace_paths.items()}
        }

        return episode_data

    def _skill_to_dict(self, skill) -> Dict:
        """Convert skill object to dictionary"""
        return {
            "name": skill.name,
            "skill_type": skill.skill_type,
            "description": skill.description,
            "preconditions": skill.preconditions,
            "effects": skill.effects,
            "instance_count": len(skill.instances)
        }

    def _analyze_cooperation(
        self,
        actions: List,
        coop_skills: List,
        tom_analysis: List
    ) -> Dict:
        """Analyze cooperation patterns in the episode"""
        # Count actions by agent
        agent_0_actions = [a for a in actions if a.agent_id == 0 and a.action_type != "observation"]
        agent_1_actions = [a for a in actions if a.agent_id == 1 and a.action_type != "observation"]

        # Count coordination actions
        agent_0_waits = sum(1 for a in agent_0_actions if a.action_type == "Wait")
        agent_1_waits = sum(1 for a in agent_1_actions if a.action_type == "Wait")

        # Determine if coordination was needed
        requires_coordination = len(coop_skills) > 0 or (agent_0_waits + agent_1_waits) > 0

        # Cooperation patterns found
        patterns = []
        for skill in coop_skills:
            patterns.append(skill.name)

        return {
            "requires_coordination": requires_coordination,
            "total_coordination_actions": agent_0_waits + agent_1_waits,
            "coordination_effective": True,  # Successful episodes by definition
            "has_coordination_patterns": len(patterns) > 0,
            "patterns": patterns,
            "agent_0_coordination": {
                "wait_actions": agent_0_waits,
                "total_actions": len(agent_0_actions)
            },
            "agent_1_coordination": {
                "wait_actions": agent_1_waits,
                "total_actions": len(agent_1_actions)
            },
            "tom_cooperation": {
                "has_tom_reasoning": len(tom_analysis) > 0,
                "tom_quality": {
                    "avg_tom_completeness": sum(t.confidence for t in tom_analysis) / len(tom_analysis) if tom_analysis else 0,
                    "total_tom_cooperation_patterns": len(set(t.cooperation_pattern for t in tom_analysis))
                }
            }
        }

    def build_dataset(self):
        """Build the complete RAG dataset"""
        print("=" * 80)
        print("Building RAG Dataset from Heuristic Trajectories")
        print("=" * 80)

        # Load episode results
        episode_results = self.load_episode_results()
        if not episode_results:
            print("No episodes found!")
            return

        # Find traces directory
        traces_base = self.find_traces_directory()
        if not traces_base:
            print(f"Error: Traces directory not found in {self.results_dir}")
            return

        print(f"Using traces from: {traces_base}")

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        traces_output = os.path.join(self.output_dir, "react_trajectories", "traces")
        os.makedirs(os.path.join(traces_output, "0"), exist_ok=True)
        os.makedirs(os.path.join(traces_output, "1"), exist_ok=True)

        # Process each episode
        processed = 0
        skipped = 0

        for episode_id, info in episode_results.items():
            # Find trace files
            trace_paths = {}
            for agent_id in [0, 1]:
                trace_pattern = f"trace-episode_{episode_id}_0-{agent_id}.txt"
                trace_path = os.path.join(traces_base, str(agent_id), trace_pattern)
                if os.path.exists(trace_path):
                    trace_paths[agent_id] = trace_path

            if not trace_paths:
                skipped += 1
                continue

            # Process episode
            episode_data = self.process_episode(episode_id, info, trace_paths)

            if episode_data:
                self.episodes.append(episode_data)
                processed += 1

                # Copy trace files
                for agent_id, src_path in trace_paths.items():
                    dst_path = os.path.join(traces_output, str(agent_id), os.path.basename(src_path))
                    shutil.copy2(src_path, dst_path)

                if processed % 50 == 0:
                    print(f"  Processed {processed}/{len(episode_results)} episodes...")
            else:
                skipped += 1

        print(f"Processed {processed} episodes, skipped {skipped}")

        # Update metadata
        self._update_metadata()

        # Save dataset
        self._save_dataset()

    def _update_metadata(self):
        """Update dataset metadata with statistics"""
        total_coop_patterns = 0
        total_tom_episodes = 0
        coordination_episodes = 0

        for ep in self.episodes:
            coop = ep.get("cooperation_skills", {})
            if coop.get("requires_coordination"):
                coordination_episodes += 1
            if coop.get("has_coordination_patterns"):
                total_coop_patterns += len(coop.get("patterns", []))

            tom = ep.get("tom_analysis", {})
            if tom.get("has_tom_reasoning"):
                total_tom_episodes += 1

        self.metadata.update({
            "total_episodes": len(self.episodes),
            "enhanced_with_cooperation_skills": True,
            "enhanced_with_tom_analysis": True,
            "cooperation_statistics": {
                "episodes_with_coordination_patterns": coordination_episodes,
                "episodes_requiring_coordination": coordination_episodes,
                "episodes_with_tom_reasoning": total_tom_episodes,
                "total_cooperation_patterns": total_coop_patterns,
                "avg_patterns_per_episode": total_coop_patterns / len(self.episodes) if self.episodes else 0
            }
        })

    def _save_dataset(self):
        """Save the dataset to disk"""
        # Save main dataset file
        dataset = {
            "metadata": self.metadata,
            "episodes": self.episodes
        }

        output_file = os.path.join(self.output_dir, "react_trajectories.json.gz")
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)

        print(f"Dataset saved to {output_file}")

        # Save episode result log
        log_file = os.path.join(self.output_dir, "episode_result_log.csv")
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "episode_id", "instruction", "success_rate",
                "task_percent_complete", "requires_coordination",
                "has_tom_reasoning", "cooperation_patterns"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ep in self.episodes:
                writer.writerow({
                    "episode_id": ep["episode_id"],
                    "instruction": ep["instruction"],
                    "success_rate": ep["success_rate"],
                    "task_percent_complete": ep["task_percent_complete"],
                    "requires_coordination": ep.get("cooperation_skills", {}).get("requires_coordination", False),
                    "has_tom_reasoning": ep.get("tom_analysis", {}).get("has_tom_reasoning", False),
                    "cooperation_patterns": ",".join(ep.get("cooperation_skills", {}).get("patterns", []))
                })

        print(f"Episode log saved to {log_file}")

        # Save summary
        summary = {
            "total_episodes": len(self.episodes),
            "successful_episodes": sum(1 for ep in self.episodes if ep["success_rate"] >= 1.0),
            "coordination_episodes": sum(1 for ep in self.episodes if ep.get("cooperation_skills", {}).get("requires_coordination")),
            "tom_episodes": sum(1 for ep in self.episodes if ep.get("tom_analysis", {}).get("has_tom_reasoning")),
            "created_at": self.metadata["created_at"],
            "source": self.metadata["source"]
        }

        summary_file = os.path.join(self.output_dir, "dataset_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {summary_file}")
        print(f"\nDataset Statistics:")
        print(f"  - Total episodes: {summary['total_episodes']}")
        print(f"  - Successful: {summary['successful_episodes']}")
        print(f"  - With coordination: {summary['coordination_episodes']}")
        print(f"  - With ToM reasoning: {summary['tom_episodes']}")


def build_rag_dataset(
    results_dir: str,
    output_dir: str,
    filter_successful: bool = True
):
    """
    Main function to build RAG dataset

    Args:
        results_dir: Path to heuristic results directory
        output_dir: Path to save the RAG dataset
        filter_successful: Only include successful episodes
    """
    builder = RAGDatasetBuilder(
        results_dir=results_dir,
        output_dir=output_dir,
        filter_successful=filter_successful
    )

    builder.build_dataset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RAG dataset from heuristic trajectories")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2025-12-30_21-10-23-heterogeneous+rerange.json/results",
        help="Path to heuristic results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/rag_datasets/hierarchical_skill_rag",
        help="Path to save the RAG dataset"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed episodes"
    )

    args = parser.parse_args()

    build_rag_dataset(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        filter_successful=not args.include_failed
    )
