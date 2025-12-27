#!/usr/bin/env python3
"""
成功轨迹保存和RAG数据集更新功能

这个模块提供了将成功的交互轨迹总结并添加到RAG数据集的功能，
用于改进后续的规划性能。
"""

import csv
import json
import os
import time
from typing import Any, Dict, Optional


class SuccessTrajectoryToRAG:
    """成功轨迹到RAG数据集的转换器"""

    def __init__(self, rag_base_dir: str, example_type: str = "summary"):
        """
        初始化转换器

        Args:
            rag_base_dir: RAG数据集的基础目录
            example_type: 示例类型 ("react", "summary", "zero_shot")
        """
        self.rag_base_dir = rag_base_dir
        self.example_type = example_type
        self.embedding_model = None

        # 确保RAG目录存在
        os.makedirs(rag_base_dir, exist_ok=True)

    def save_success_trajectory_to_rag(
        self, config, env_interface, eval_runner, success_metrics: Dict[str, Any]
    ) -> Optional[str]:
        """
        保存成功轨迹并添加到RAG数据集

        Args:
            config: 配置对象
            env_interface: 环境接口对象
            eval_runner: 评估运行器对象
            success_metrics: 成功执行的指标

        Returns:
            保存的文件路径，失败时返回None
        """
        try:
            dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
            episode_id = env_interface.env.env.env._env.current_episode.episode_id
            instruction = env_interface.env.env.env._env.current_episode.instruction

            # 创建成功轨迹目录
            success_dir = os.path.join(
                config.paths.results_dir,
                dataset_file,
                "success_trajectories",
                f"episode_{episode_id}",
            )
            os.makedirs(success_dir, exist_ok=True)

            # 1. 保存基本成功信息
            success_metadata = {
                "episode_id": episode_id,
                "instruction": instruction,
                "success_timestamp": time.time(),
                "success_metrics": success_metrics,
                "trajectory_length": getattr(env_interface, "_trajectory_idx", 0),
                "success": True,
            }

            with open(os.path.join(success_dir, "success_metadata.json"), "w") as f:
                json.dump(success_metadata, f, indent=2)

            # 2. 总结轨迹内容
            trajectory_summary = self._summarize_trajectory(
                env_interface, eval_runner, instruction, episode_id
            )

            # 3. 添加到RAG数据集
            rag_entry_path = self._add_to_rag_dataset(
                episode_id, instruction, trajectory_summary, success_metrics
            )

            # 4. 更新RAG索引
            self._update_rag_index(rag_entry_path)

            print(f"成功轨迹已保存并添加到RAG数据集: {rag_entry_path}")
            return rag_entry_path

        except Exception as e:
            print(f"保存成功轨迹到RAG时发生错误: {e}")
            return None

    def _summarize_trajectory(
        self, env_interface, eval_runner, instruction: str, episode_id: int
    ) -> Dict[str, Any]:
        """总结轨迹内容"""
        summary = {
            "instruction": instruction,
            "episode_id": episode_id,
            "agent_actions": {},
            "key_states": [],
            "success_pattern": "",
        }

        # 提取智能体动作序列
        if (
            hasattr(env_interface, "agent_action_history")
            and env_interface.agent_action_history
        ):
            for agent_id, actions in env_interface.agent_action_history.items():
                action_sequence = []
                for action in actions:
                    action_summary = {
                        "type": action.action
                        if hasattr(action, "action")
                        else str(action),
                        "params": getattr(action, "params", ""),
                        "result": getattr(action, "result", "success"),
                    }
                    action_sequence.append(action_summary)
                summary["agent_actions"][agent_id] = action_sequence

        # 根据example_type格式化轨迹
        if self.example_type == "summary":
            summary["success_pattern"] = self._format_summary_pattern(summary)
        elif self.example_type == "react":
            summary["success_pattern"] = self._format_react_pattern(summary)
        elif self.example_type == "zero_shot":
            summary["success_pattern"] = self._format_zero_shot_pattern(summary)

        return summary

    def _format_summary_pattern(self, summary: Dict[str, Any]) -> str:
        """格式化为summary模式的轨迹"""
        pattern = f"Task:\n{summary['instruction']}\n\n"

        # 添加动作序列概述
        pattern += "Successful execution pattern:\n"
        for agent_id, actions in summary["agent_actions"].items():
            pattern += f"Agent {agent_id}:\n"
            for i, action in enumerate(actions):
                pattern += f"  {i+1}. {action['type']}[{action['params']}] -> {action['result']}\n"

        pattern += "\nAssigned!"
        return pattern

    def _format_react_pattern(self, summary: Dict[str, Any]) -> str:
        """格式化为react模式的轨迹"""
        pattern = f"Task: {summary['instruction']}\n\n"

        # 模拟react推理过程
        pattern += "Thought: I need to complete this task step by step.\n"
        for agent_id, actions in summary["agent_actions"].items():
            for action in actions:
                pattern += (
                    f"Agent_{agent_id}_Action: {action['type']}[{action['params']}]\n"
                )
                pattern += f"Agent_{agent_id}_Observation: {action['result']}\n"

        pattern += (
            "Final Thought: All objects were successfully moved, so I am done!\nExit!"
        )
        return pattern

    def _format_zero_shot_pattern(self, summary: Dict[str, Any]) -> str:
        """格式化为zero_shot模式的轨迹"""
        pattern = f"Task: {summary['instruction']}\n\n"

        # 直接动作序列
        for _agent_id, actions in summary["agent_actions"].items():
            for action in actions:
                pattern += f"{action['type']}[{action['params']}]\n"

        return pattern

    def _add_to_rag_dataset(
        self,
        episode_id: int,
        instruction: str,
        trajectory_summary: Dict[str, Any],
        success_metrics: Dict[str, Any],
    ) -> str:
        """将轨迹添加到RAG数据集"""

        # 创建RAG数据集目录结构
        rag_dataset_dir = os.path.join(self.rag_base_dir, "new_trajectories")
        os.makedirs(rag_dataset_dir, exist_ok=True)

        traces_dir = os.path.join(rag_dataset_dir, "traces", "0")
        os.makedirs(traces_dir, exist_ok=True)

        # 保存轨迹文件
        trace_filename = f"trace-episode_{episode_id}_0-0.txt"
        trace_path = os.path.join(traces_dir, trace_filename)

        with open(trace_path, "w") as f:
            f.write(trajectory_summary["success_pattern"])

        # 更新或创建episode_result_log.csv
        csv_path = os.path.join(rag_dataset_dir, "episode_result_log.csv")

        # 检查CSV是否存在，如果不存在则创建表头
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            if not file_exists:
                # 写入表头
                writer.writerow(
                    ["episode_id", "success_rate", "instruction", "success"]
                )

            # 写入成功记录
            writer.writerow([episode_id, 1.0, instruction, 1.0])

        return trace_path

    def _update_rag_index(self, new_trace_path: str):
        """更新RAG索引以包含新轨迹"""
        # 这里可以实现增量更新RAG嵌入索引的逻辑
        # 为了简化，目前只记录需要重建索引的标志
        index_flag_path = os.path.join(self.rag_base_dir, ".needs_reindex")
        with open(index_flag_path, "w") as f:
            f.write(f"New trajectory added: {new_trace_path}\n")

        print("RAG索引标记为需要更新")


class DynamicRAGUpdater:
    """动态RAG更新器，支持实时更新RAG数据集"""

    def __init__(self, rag_instance):
        """
        初始化动态更新器

        Args:
            rag_instance: 现有的RAG实例
        """
        self.rag = rag_instance

    def add_new_trajectory_to_existing_rag(
        self, episode_id: int, instruction: str, trace_content: str, agent_id: int = 0
    ):
        """
        动态添加新轨迹到现有RAG实例

        Args:
            episode_id: episode ID
            instruction: 指令内容
            trace_content: 轨迹内容
            agent_id: 智能体ID
        """
        # 创建新的数据条目
        new_index = len(self.rag.data_dict)

        new_info = {
            "instruction": instruction,
            "trace": trace_content,
            "agent_id": agent_id,
            "file": f"dynamic_episode_{episode_id}.txt",
            "episode_id": episode_id,
            "timestamp": time.time(),
        }

        # 计算新轨迹的嵌入
        if hasattr(self.rag, "embedding_model"):
            new_embedding = self.rag.embedding_model.encode(
                instruction, convert_to_tensor=True
            )
            new_info["embedding"] = new_embedding

        # 添加到数据字典
        self.rag.data_dict[new_index] = new_info
        self.rag.index = new_index + 1

        print(f"成功添加新轨迹到RAG: Episode {episode_id}")

    def save_updated_rag_dataset(self, output_dir: str):
        """保存更新后的RAG数据集"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存所有轨迹
        for _index, info in self.rag.data_dict.items():
            if "episode_id" in info:  # 动态添加的轨迹
                trace_file = os.path.join(
                    output_dir, f"trace-episode_{info['episode_id']}.txt"
                )
                with open(trace_file, "w") as f:
                    f.write(info["trace"])

        # 更新CSV文件
        csv_file = os.path.join(output_dir, "episode_result_log.csv")
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["episode_id", "success_rate", "instruction", "success"])

            for info in self.rag.data_dict.values():
                if "episode_id" in info:
                    writer.writerow([info["episode_id"], 1.0, info["instruction"], 1.0])

        print(f"更新后的RAG数据集已保存到: {output_dir}")


# 使用示例和集成函数
def integrate_success_trajectory_saving(config, env_interface, eval_runner):
    """集成成功轨迹保存功能到现有系统"""

    # 初始化成功轨迹保存器
    rag_base_dir = config.get("rag_base_dir", "data/rag_datasets/")
    example_type = config.get("example_type", "summary")

    trajectory_saver = SuccessTrajectoryToRAG(rag_base_dir, example_type)

    # 在planner_demo.py的成功处理部分添加
    def enhanced_save_success_message(config, env_interface, success_metrics):
        """增强的成功消息保存，包含RAG更新"""

        # 原有的成功保存逻辑
        output_file = get_output_file(config, env_interface)
        success_dict = {"success": True, "stats": json.dumps(success_metrics)}
        with open(output_file, "w+") as f:
            f.write(json.dumps(success_dict))

        # 新增：保存到RAG数据集
        if config.evaluation.get("save_to_rag", False):
            trajectory_saver.save_success_trajectory_to_rag(
                config, env_interface, eval_runner, success_metrics
            )

    return enhanced_save_success_message


def get_output_file(config, env_interface):
    """获取输出文件路径（简化版本）"""
    env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    return os.path.join(config.paths.results_dir, f"episode_{episode_id}_result.json")
