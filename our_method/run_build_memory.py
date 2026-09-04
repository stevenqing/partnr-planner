#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to build hierarchical skill memory and RAG dataset from heuristic results

Usage:
    # Rule-based extraction:
    python run_build_memory.py --results-dir <path> --output-dir <path>

    # LLM-based extraction with Llama 3.3 70B:
    python run_build_memory.py --results-dir <path> --use-llm --vllm-host <host> --vllm-port <port>

This script:
1. Builds the hierarchical skill memory (L_ind, L_coop)
2. Builds the RAG dataset with cooperation skills and ToM analysis
3. Saves all outputs for use with the planner

Reference: /doc/our_method
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_hierarchical_skill_memory import build_memory_from_heuristic_results
from build_rag_dataset import build_rag_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical skill memory and RAG dataset from heuristic results"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm/2025-12-30_21-10-23-heterogeneous+rerange.json/results",
        help="Path to heuristic results directory"
    )

    parser.add_argument(
        "--memory-output-dir",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/hierarchical_skill_memory",
        help="Path to save the hierarchical skill memory"
    )

    parser.add_argument(
        "--rag-output-dir",
        type=str,
        default="/home/a5l/shuqing.a5l/partnr-planner/data/rag_datasets/hierarchical_skill_rag",
        help="Path to save the RAG dataset"
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed episodes in memory and dataset"
    )

    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip building hierarchical memory (only build RAG dataset)"
    )

    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip building RAG dataset (only build hierarchical memory)"
    )

    # LLM options
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Llama 3.3 70B for skill extraction (requires vLLM server)"
    )

    parser.add_argument(
        "--vllm-host",
        type=str,
        default="localhost",
        help="vLLM server host"
    )

    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="vLLM server port"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for vLLM"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Hierarchical Skill Memory and RAG Dataset Builder")
    print("=" * 80)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Memory output: {args.memory_output_dir}")
    print(f"RAG output: {args.rag_output_dir}")
    print(f"Include failed: {args.include_failed}")
    if args.use_llm:
        print(f"\nLLM Extraction: ENABLED")
        print(f"  Model: {args.model_name}")
        print(f"  vLLM Server: {args.vllm_host}:{args.vllm_port}")
    else:
        print(f"\nLLM Extraction: DISABLED (using rule-based extraction)")
    print()

    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Step 1: Build hierarchical skill memory
    if not args.skip_memory:
        print("\n" + "=" * 40)
        print("Step 1: Building Hierarchical Skill Memory")
        print("=" * 40)

        memory = build_memory_from_heuristic_results(
            results_dir=args.results_dir,
            output_dir=args.memory_output_dir,
            filter_successful=not args.include_failed,
            use_llm=args.use_llm,
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            model_name=args.model_name
        )

        if memory:
            print(f"\nHierarchical memory saved to: {args.memory_output_dir}")
        else:
            print("Warning: Memory building may have encountered issues")

    # Step 2: Build RAG dataset
    if not args.skip_rag:
        print("\n" + "=" * 40)
        print("Step 2: Building RAG Dataset")
        print("=" * 40)

        build_rag_dataset(
            results_dir=args.results_dir,
            output_dir=args.rag_output_dir,
            filter_successful=not args.include_failed
        )

        print(f"\nRAG dataset saved to: {args.rag_output_dir}")

    print("\n" + "=" * 80)
    print("Build Complete!")
    print("=" * 80)

    # Print usage instructions
    print("\nTo use the hierarchical skill memory with the planner:")
    print("```python")
    print("from our_method.planner_integration import create_planner")
    print(f"planner = create_planner('{args.memory_output_dir}')")
    print("```")

    print("\nTo use RAG integration:")
    print("```python")
    print("from our_method.planner_integration import create_rag_integration")
    print(f"rag = create_rag_integration('{args.memory_output_dir}', '{args.rag_output_dir}')")
    print("examples = rag.get_examples(instruction, world_state)")
    print("```")


if __name__ == "__main__":
    main()
