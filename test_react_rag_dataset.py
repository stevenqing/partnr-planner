#!/usr/bin/env python3
"""
测试React格式RAG数据集是否可以正常加载

使用方法:
python test_react_rag_dataset.py --rag-dir data/rag_datasets/react_rag_dataset_fixed
"""

import argparse
import os
import sys
from types import SimpleNamespace

# 添加habitat_llm到路径
sys.path.append(".")


def test_react_rag_dataset(rag_dir: str, data_source_name: str = "react_trajectories"):
    """测试React格式RAG数据集加载"""
    try:
        from habitat_llm.planner.rag import RAG

        print(f"测试React RAG数据集: {rag_dir}")
        print(f"数据源名称: {data_source_name}")
        print("示例类型: react")

        # 模拟LLM配置
        _llm_config = dict(
            system_tag="<|start_header_id|>system<|end_header_id|>\n",
            user_tag="<|start_header_id|>user<|end_header_id|>\n",
            assistant_tag="<|start_header_id|>assistant<|end_header_id|>\n",
            eot_tag="<|eot_id|>\n",
        )
        llm_config = SimpleNamespace(**_llm_config)

        # 初始化RAG
        # 确保路径以斜杠结尾
        if not rag_dir.endswith("/"):
            rag_dir = rag_dir + "/"

        data_dir = [rag_dir]
        data_source_name_list = [data_source_name]
        example_type = "react"

        print("正在初始化RAG...")
        rag = RAG(example_type, data_dir, data_source_name_list, llm_config)

        print("✅ RAG初始化成功！")
        print(f"   数据字典大小: {len(rag.data_dict)}")
        print(f"   起始头部索引: {rag.start_header_idx}")

        # 显示加载的轨迹信息
        for idx, data in rag.data_dict.items():
            print(f"   轨迹 {idx}: {data['instruction'][:60]}...")
            print(f"     文件: {os.path.basename(data.get('file', 'N/A'))}")
            print(f"     轨迹长度: {len(data.get('trace', ''))}")

        # 测试检索功能
        test_instruction = "Move the apple to the table"
        print("\n测试检索功能...")
        print(f"查询指令: {test_instruction}")

        scores, indices = rag.retrieve_top_k_given_query(
            test_instruction, top_k=1, agent_id=0
        )

        if len(scores) > 0:
            print("✅ 检索成功！")
            print(f"   最相似轨迹索引: {indices[0]}")
            print(f"   相似度分数: {scores[0]:.4f}")

            # 显示检索到的轨迹预览
            retrieved_trace = rag.data_dict[indices[0]]["trace"]
            print("   检索到的轨迹预览:")
            preview = (
                retrieved_trace[:300] + "..."
                if len(retrieved_trace) > 300
                else retrieved_trace
            )
            print(f"   {preview}")
        else:
            print("⚠ 检索未找到结果")

        return True

    except Exception as e:
        print(f"❌ RAG测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试React RAG数据集")
    parser.add_argument(
        "--rag-dir",
        default="data/rag_datasets/react_rag_dataset_fixed",
        help="RAG数据集目录",
    )
    parser.add_argument(
        "--data-source-name", default="react_trajectories", help="数据源名称"
    )

    args = parser.parse_args()

    if not os.path.exists(args.rag_dir):
        print(f"❌ RAG数据集目录不存在: {args.rag_dir}")
        return 1

    success = test_react_rag_dataset(args.rag_dir, args.data_source_name)

    if success:
        print("\n🎉 测试通过！React RAG数据集可以正常使用")
        print("\n下一步可以运行:")
        print("./run_planner_demo_with_rag.sh")
        return 0
    else:
        print("\n💥 测试失败，需要检查数据集格式")
        return 1


if __name__ == "__main__":
    exit(main())
