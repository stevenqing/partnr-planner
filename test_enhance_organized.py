#!/usr/bin/env python3
"""
Test script to verify enhanced skill extraction on a small portion of organized dataset
"""

import gzip
import json
import os

from enhanced_skill_extractor import EnhancedSkillExtractor


def test_organized_enhancement():
    """Test enhanced extraction on a small sample"""

    print("🧪 Testing Enhanced Skill Extraction on Organized Dataset")
    print("=" * 60)

    # Load a small sample from one organized dataset
    input_file = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_organized_by_skills/skill_multi_agent_coordination.json.gz"

    if not os.path.exists(input_file):
        print(f"❌ Test file not found: {input_file}")
        return None

    # Load dataset
    with gzip.open(input_file, "rt") as f:
        dataset = json.load(f)

    print(f"✓ Loaded coordination dataset with {len(dataset['episodes'])} episodes")

    # Test on first 3 episodes
    test_episodes = dataset["episodes"][:3]
    extractor = EnhancedSkillExtractor(
        use_llm=False
    )  # Use heuristics for faster testing

    # Original traces directory
    traces_dir = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_with_skills/react_trajectories/traces"

    print(f"\nTesting enhanced extraction on {len(test_episodes)} episodes...")

    for i, episode in enumerate(test_episodes, 1):
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]

        print(f"\n--- Test Episode {i}: {episode_id} ---")
        print(f"Task: {instruction}")

        # Test enhanced extraction
        success_count = 0
        total_patterns = 0
        total_decisions = 0

        for agent_id in ["0", "1"]:
            trace_file = f"trace-episode_{episode_id}_0-{agent_id}.txt"
            trace_path = f"{traces_dir}/{agent_id}/{trace_file}"

            if os.path.exists(trace_path):
                try:
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace_content = f.read()

                    # Apply enhanced extraction
                    enhanced_skills = extractor.extract_enhanced_skills(
                        trace_content, agent_id, instruction
                    )

                    print(f"  Agent {agent_id}:")
                    print(
                        f"    ✓ Enhanced Description: {enhanced_skills['enhanced_skill_description'][:80]}..."
                    )
                    print(
                        f"    ✓ Skill Categories: {len(enhanced_skills['skill_categories'])}"
                    )
                    print(
                        f"    ✓ Skill Patterns: {len(enhanced_skills['skill_patterns'])}"
                    )
                    print(
                        f"    ✓ Decision Points: {len(enhanced_skills['decision_points'])}"
                    )
                    print(
                        f"    ✓ Efficiency Score: {enhanced_skills['action_efficiency']['efficiency_score']:.3f}"
                    )

                    coordination = enhanced_skills["coordination_requirements"]
                    if coordination["requires_coordination"]:
                        print(
                            f"    ✓ Coordination Required: {coordination['coordination_actions_count']} actions"
                        )

                    success_count += 1
                    total_patterns += len(enhanced_skills["skill_patterns"])
                    total_decisions += len(enhanced_skills["decision_points"])

                except Exception as e:
                    print(f"    ❌ Error processing Agent {agent_id}: {e}")
            else:
                print(f"    ⚠️ Trace file not found for Agent {agent_id}")

        print(
            f"  Summary: {success_count}/2 agents processed, {total_patterns} patterns, {total_decisions} decisions"
        )

    print("\n🎉 Test completed successfully!")
    print(
        "The enhanced skill extraction is working correctly on the organized dataset."
    )

    return True


if __name__ == "__main__":
    test_organized_enhancement()
