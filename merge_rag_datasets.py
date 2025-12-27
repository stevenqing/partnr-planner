#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import shutil
from collections import Counter
from datetime import datetime


def merge_rag_datasets():
    """合并spatial_only和rerange_only_cleaned两个RAG数据集"""

    # 输入数据集路径
    spatial_dir = "data/rag_datasets/spatial_only"
    rerange_dir = "data/rag_datasets/rerange_only_cleaned"

    # 输出路径
    output_dir = "data/rag_datasets/spatial_rerange_merged"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("=" * 80)
    print("合并RAG数据集")
    print("=" * 80)
    print(f"源数据集1: {spatial_dir}")
    print(f"源数据集2: {rerange_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 读取spatial_only数据集
    print("📖 读取spatial_only数据集...")
    spatial_json = f"{spatial_dir}/react_trajectories.json.gz"
    with gzip.open(spatial_json, "rt", encoding="utf-8") as f:
        spatial_data = json.load(f)

    spatial_episodes = spatial_data.get("episodes", [])
    spatial_metadata = spatial_data.get("metadata", {})
    print(f"  ✓ 读取 {len(spatial_episodes)} 个episodes")

    # 读取rerange_only_cleaned数据集
    print("📖 读取rerange_only_cleaned数据集...")
    rerange_json = f"{rerange_dir}/react_trajectories.json.gz"
    with gzip.open(rerange_json, "rt", encoding="utf-8") as f:
        rerange_data = json.load(f)

    rerange_episodes = rerange_data.get("episodes", [])
    rerange_metadata = rerange_data.get("metadata", {})
    print(f"  ✓ 读取 {len(rerange_episodes)} 个episodes")

    # 检查episode_id冲突（保持原始ID，因为RAG加载器需要整数ID）
    print("\n🔄 检查episode_id冲突...")
    spatial_episode_ids = set()
    rerange_episode_ids = set()

    for ep in spatial_episodes:
        original_id = ep.get("episode_id", "")
        spatial_episode_ids.add(original_id)
        # 保持原始ID，但添加source_dataset标记
        ep["source_dataset"] = "spatial_only"

    for ep in rerange_episodes:
        original_id = ep.get("episode_id", "")
        rerange_episode_ids.add(original_id)
        # 保持原始ID，但添加source_dataset标记
        ep["source_dataset"] = "rerange_only_cleaned"

    conflicts = spatial_episode_ids & rerange_episode_ids
    if conflicts:
        print(f"  ⚠️  发现 {len(conflicts)} 个episode_id冲突，需要重新编号")
        # 为spatial_only的冲突ID重新编号（从10000开始）
        conflict_map = {}
        new_id = 10000
        for ep in spatial_episodes:
            original_id = ep.get("episode_id", "")
            if original_id in conflicts:
                if original_id not in conflict_map:
                    conflict_map[original_id] = str(new_id)
                    new_id += 1
                ep["episode_id"] = conflict_map[original_id]
                ep["original_episode_id"] = original_id
        print(f"  ✓ 已重新编号 {len(conflict_map)} 个冲突的episode_id")
    else:
        print("  ✓ 无episode_id冲突，保持原始ID")

    # 合并episodes
    print("\n📦 合并episodes...")
    merged_episodes = spatial_episodes + rerange_episodes
    print(f"  ✓ 合并后共 {len(merged_episodes)} 个episodes")

    # 合并元数据统计
    print("\n📊 合并元数据统计...")

    # 合并技能统计
    spatial_stats = spatial_metadata.get("skill_statistics", {})
    rerange_stats = rerange_metadata.get("skill_statistics", {})

    # 合并任务类型分布
    task_type_dist = Counter(spatial_stats.get("task_type_distribution", {}))
    task_type_dist.update(rerange_stats.get("task_type_distribution", {}))

    # 合并复杂度分布
    complexity_dist = Counter(spatial_stats.get("complexity_distribution", {}))
    complexity_dist.update(rerange_stats.get("complexity_distribution", {}))

    # 合并技能类别分布
    skill_category_dist = Counter(spatial_stats.get("skill_category_distribution", {}))
    skill_category_dist.update(rerange_stats.get("skill_category_distribution", {}))

    merged_skill_statistics = {
        "task_type_distribution": dict(task_type_dist),
        "complexity_distribution": dict(complexity_dist),
        "skill_category_distribution": dict(skill_category_dist),
        "total_episodes_with_skills": (
            spatial_stats.get("total_episodes_with_skills", 0)
            + rerange_stats.get("total_episodes_with_skills", 0)
        ),
    }

    # 创建合并后的元数据
    merged_metadata = {
        "total_episodes": len(merged_episodes),
        "source": f"merged from {spatial_metadata.get('source', 'spatial_only')} and {rerange_metadata.get('source', 'rerange_only_cleaned')}",
        "created_by": "merge_rag_datasets.py",
        "example_type": "react",
        "description": f"合并数据集: spatial_only ({len(spatial_episodes)} episodes) + rerange_only_cleaned ({len(rerange_episodes)} episodes)",
        "enhanced_with_skills": True,
        "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
        "enhanced_by": "merge_rag_datasets.py",
        "skill_model": "Llama 3.1 8B",
        "skill_statistics": merged_skill_statistics,
        "source_datasets": {
            "spatial_only": {
                "episodes": len(spatial_episodes),
                "source": spatial_metadata.get("source", ""),
                "enhancement_date": spatial_metadata.get("enhancement_date", ""),
            },
            "rerange_only_cleaned": {
                "episodes": len(rerange_episodes),
                "source": rerange_metadata.get("source", ""),
                "enhancement_date": rerange_metadata.get("enhancement_date", ""),
            },
        },
    }

    print(f"  ✓ 任务类型分布: {len(merged_skill_statistics['task_type_distribution'])} 种")
    print(f"  ✓ 复杂度分布: {len(merged_skill_statistics['complexity_distribution'])} 种")
    print(
        f"  ✓ 技能类别分布: {len(merged_skill_statistics['skill_category_distribution'])} 种"
    )

    # 复制trace文件
    print("\n📁 复制trace文件...")
    trace_count = 0

    # 复制spatial_only的trace文件
    spatial_traces_dir = f"{spatial_dir}/react_trajectories/traces"
    for agent_id in ["0", "1"]:
        source_agent_dir = f"{spatial_traces_dir}/{agent_id}"
        target_agent_dir = f"{output_dir}/react_trajectories/traces/{agent_id}"

        if os.path.exists(source_agent_dir):
            for filename in os.listdir(source_agent_dir):
                if filename.startswith("trace-episode_") and filename.endswith(".txt"):
                    # 保持原始文件名格式（RAG加载器期望原始格式）
                    source_path = f"{source_agent_dir}/{filename}"
                    target_path = f"{target_agent_dir}/{filename}"
                    shutil.copy2(source_path, target_path)
                    trace_count += 1

    # 复制rerange_only_cleaned的trace文件
    rerange_traces_dir = f"{rerange_dir}/react_trajectories/traces"
    for agent_id in ["0", "1"]:
        source_agent_dir = f"{rerange_traces_dir}/{agent_id}"
        target_agent_dir = f"{output_dir}/react_trajectories/traces/{agent_id}"

        if os.path.exists(source_agent_dir):
            for filename in os.listdir(source_agent_dir):
                if filename.startswith("trace-episode_") and filename.endswith(".txt"):
                    # 保持原始文件名格式（RAG加载器期望原始格式）
                    source_path = f"{source_agent_dir}/{filename}"
                    target_path = f"{target_agent_dir}/{filename}"
                    shutil.copy2(source_path, target_path)
                    trace_count += 1

    print(f"  ✓ 复制了 {trace_count} 个trace文件")

    # 创建主JSON文件
    print("\n💾 保存合并后的数据集...")
    merged_data = {"metadata": merged_metadata, "episodes": merged_episodes}

    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存JSON文件: {output_json_path}")

    # 创建episode_result_log.csv（使用空格分隔符，与RAG加载器兼容）
    csv_path = f"{output_dir}/episode_result_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        # RAG加载器期望格式: episode_id instruction success
        # 使用空格分隔符，instruction用引号包围
        f.write("episode_id instruction success\n")
        valid_count = 0
        invalid_count = 0

        for ep in merged_episodes:
            episode_id = str(ep.get("episode_id", ""))

            # 清理episode_id：如果包含trace文件名模式，提取数字部分
            # 例如: "1024_0-1.txt" -> "1024"
            if "_0-0.txt" in episode_id or "_0-1.txt" in episode_id:
                # 提取数字部分
                import re

                match = re.search(r"^(\d+)_", episode_id)
                if match:
                    episode_id = match.group(1)
                    invalid_count += 1
                else:
                    # 如果无法提取，跳过这一行
                    continue

            instruction = ep.get("instruction", "").replace('"', '""')  # 转义引号
            success = ep.get("success_rate", 0)
            # 格式: episode_id "instruction" success
            f.write(f'{episode_id} "{instruction}" {success}\n')
            valid_count += 1

        if invalid_count > 0:
            print(f"  ⚠️  清理了 {invalid_count} 个包含trace文件名的episode_id")
        print(f"  ✓ 保存CSV文件: {csv_path} ({valid_count} 个episodes)")

    # 打印总结
    print("\n" + "=" * 80)
    print("✅ 合并完成!")
    print("=" * 80)
    print(f"总episodes数: {len(merged_episodes)}")
    print(f"  - spatial_only: {len(spatial_episodes)}")
    print(f"  - rerange_only_cleaned: {len(rerange_episodes)}")
    print(f"总trace文件数: {trace_count}")
    print(f"输出目录: {output_dir}")
    print(f"主文件: {output_json_path}")
    print(f"CSV文件: {csv_path}")
    print("=" * 80)

    return output_dir


if __name__ == "__main__":
    output_dir = merge_rag_datasets()
    print(f"\n合并完成！输出目录: {output_dir}")
