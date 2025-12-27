#!/usr/bin/env python3
"""
脚本：将环境描述从rerange_only_converted数据集添加到rerange_only_skills_dataset数据集

这个脚本会：
1. 读取rerange_only_converted数据集中的轨迹文件
2. 提取其中的环境描述（Furniture和Objects部分）
3. 将这些描述添加到rerange_only_skills_dataset数据集的对应文件中
"""

from pathlib import Path


def extract_environment_info(trace_file_path):
    """从轨迹文件中提取环境信息（Furniture和Objects部分）"""
    try:
        with open(trace_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取Furniture部分
        furniture_start = content.find("Furniture:")
        furniture_end = content.find("Objects:")

        furniture_section = ""
        if furniture_start != -1 and furniture_end != -1:
            furniture_section = content[furniture_start:furniture_end].strip()

        # 提取Objects部分
        objects_start = content.find("Objects:")
        thought_start = content.find("Thought:")

        objects_section = ""
        if objects_start != -1 and thought_start != -1:
            objects_section = content[objects_start:thought_start].strip()

        return furniture_section, objects_section
    except Exception as e:
        print(f"读取文件 {trace_file_path} 时出错: {e}")
        return "", ""


def add_environment_to_skills_trace(
    skills_trace_path, furniture_section, objects_section
):
    """将环境信息添加到skills轨迹文件中"""
    try:
        with open(skills_trace_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到Task行之后的位置
        task_end = content.find("\n", content.find("Task:"))
        if task_end == -1:
            task_end = 0

        # 构建新的内容
        new_content = content[: task_end + 1]

        # 添加环境信息
        if furniture_section:
            new_content += furniture_section + "\n"
        if objects_section:
            new_content += objects_section + "\n"

        # 添加剩余内容（从Skill Summary开始）
        skill_summary_start = content.find("Skill Summary:")
        if skill_summary_start != -1:
            new_content += content[skill_summary_start:]

        # 写回文件
        with open(skills_trace_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"更新文件 {skills_trace_path} 时出错: {e}")
        return False


def process_dataset():
    """处理整个数据集"""
    base_dir = Path("data/rag_datasets")
    source_dir = base_dir / "rerange_only_converted" / "react_trajectories" / "traces"
    target_dir = (
        base_dir / "rerange_only_skills_dataset" / "react_trajectories" / "traces"
    )

    processed_count = 0
    error_count = 0

    # 遍历所有agent目录
    for agent_dir in source_dir.iterdir():
        if agent_dir.is_dir():
            agent_id = agent_dir.name
            target_agent_dir = target_dir / agent_id

            if not target_agent_dir.exists():
                print(f"目标目录不存在: {target_agent_dir}")
                continue

            print(f"处理agent {agent_id}...")

            # 遍历所有轨迹文件
            for trace_file in agent_dir.glob("trace-episode_*.txt"):
                episode_id = trace_file.stem
                target_trace_file = target_agent_dir / f"{episode_id}.txt"

                if not target_trace_file.exists():
                    print(f"目标文件不存在: {target_trace_file}")
                    continue

                # 提取环境信息
                furniture_section, objects_section = extract_environment_info(
                    trace_file
                )

                if furniture_section or objects_section:
                    # 添加到skills轨迹文件
                    if add_environment_to_skills_trace(
                        target_trace_file, furniture_section, objects_section
                    ):
                        processed_count += 1
                        print(f"✓ 已更新: {episode_id}")
                    else:
                        error_count += 1
                        print(f"✗ 更新失败: {episode_id}")
                else:
                    print(f"- 无环境信息: {episode_id}")

    print("\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")


if __name__ == "__main__":
    process_dataset()
