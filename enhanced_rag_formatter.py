# 智能RAG示例格式化，利用enhanced_skills数据


def format_enhanced_rag_examples(rag_data_dict, selected_indices):
    """
    将检索到的enhanced skills数据格式化为prompt中的例子
    """
    formatted_examples = []

    for idx in selected_indices:
        episode = rag_data_dict[idx]

        # 构建技能分析部分
        skill_analysis = _build_skill_analysis_section(episode)

        # 构建示例格式
        example = f"""
## Example {len(formatted_examples) + 1}: {episode.get('task_type', 'Task')} (Quality: {episode.get('quality_score', 0):.1f})

**Task**: {episode['instruction']}

{skill_analysis}

**Key Learning Points**:
{_extract_learning_insights(episode)}

**Successful Execution**:
{episode['trace']}

---
"""
        formatted_examples.append(example)

    return "\n".join(formatted_examples)


def _build_skill_analysis_section(episode):
    """构建技能分析部分"""
    analysis = "**Skill Pattern Analysis**:\n"

    # 技能模式
    if episode.get("skill_patterns"):
        analysis += f"- Patterns Used: {', '.join(episode['skill_patterns'])}\n"

    # 决策复杂度
    if episode.get("decision_count", 0) > 0:
        analysis += (
            f"- Decision Points: {episode['decision_count']} strategic decisions\n"
        )

    # 协作需求
    if episode.get("coordination_required"):
        analysis += "- Coordination: Multi-agent collaboration required\n"
    else:
        analysis += "- Coordination: Independent execution\n"

    # 效率指标
    if episode.get("efficiency_score"):
        analysis += f"- Efficiency: {episode['efficiency_score']:.1%} success rate\n"

    analysis += f"- Complexity: {episode.get('complexity', 'Unknown')}\n"

    return analysis


def _extract_learning_insights(episode):
    """提取学习要点"""
    insights = []

    # 基于技能模式的insights
    patterns = episode.get("skill_patterns", [])
    if "pick_and_place_sequence" in patterns:
        insights.append(
            "• Sequential manipulation: Pick then place in one fluid motion"
        )
    if "obstacle_avoidance" in patterns:
        insights.append("• Navigation: Identify and avoid obstacles efficiently")
    if "coordination_handoff" in patterns:
        insights.append("• Teamwork: Coordinate object handoffs between agents")

    # 基于复杂度的insights
    complexity = episode.get("complexity", "")
    if complexity == "High":
        insights.append("• Complex tasks require breaking down into smaller sub-goals")
    elif complexity == "Medium":
        insights.append("• Medium tasks benefit from systematic approach")

    # 基于协作的insights
    if episode.get("coordination_required"):
        insights.append(
            "• Multi-agent coordination prevents conflicts and improves efficiency"
        )

    # 基于效率的insights
    if episode.get("efficiency_score", 0) > 0.9:
        insights.append("• High efficiency achieved through minimal redundant actions")

    return (
        "\n".join(insights)
        if insights
        else "• Apply systematic approach to task completion"
    )


# 示例使用
def create_enhanced_rag_prompt(query, rag_system, agent_id=0):
    """
    创建包含enhanced skills分析的RAG prompt
    """

    # 智能检索：结合技能模式、质量和协作需求
    scores, indices = rag_system.retrieve_top_k_with_filters(
        query=query,
        top_k=5,
        agent_id=agent_id,
        skill_filter=["navigation", "manipulation", "coordination"],  # 根据任务类型调整
        min_quality=0.7,
        coordination_required=None,  # None表示不过滤
    )

    # 格式化为增强的示例
    rag_examples = format_enhanced_rag_examples(rag_system.data_dict, indices)

    return rag_examples


# 高级检索策略
def adaptive_skill_retrieval(query, rag_system, agent_id=0):
    """
    根据查询内容自适应选择检索策略
    """

    # 分析查询内容，确定技能需求
    skill_requirements = _analyze_query_skills(query)

    # 基于技能需求调整检索参数
    if "coordination" in skill_requirements:
        # 协作任务：优先协作示例
        scores, indices = rag_system.retrieve_top_k_with_filters(
            query=query,
            top_k=8,
            agent_id=agent_id,
            coordination_required=True,
            min_quality=0.8,
        )
    elif "navigation" in skill_requirements:
        # 导航任务：优先导航技能模式
        scores, indices = rag_system.retrieve_top_k_with_filters(
            query=query,
            top_k=6,
            agent_id=agent_id,
            skill_filter=["navigation", "obstacle_avoidance"],
            min_quality=0.6,
        )
    else:
        # 通用任务：平衡检索
        scores, indices = rag_system.retrieve_top_k_with_filters(
            query=query, top_k=5, agent_id=agent_id, min_quality=0.7
        )

    return create_enhanced_rag_prompt(query, rag_system, agent_id)


def _analyze_query_skills(query):
    """分析查询内容，识别技能需求"""
    skills = []

    if any(word in query.lower() for word in ["move", "navigate", "go to", "explore"]):
        skills.append("navigation")
    if any(word in query.lower() for word in ["pick", "place", "put", "grab"]):
        skills.append("manipulation")
    if any(word in query.lower() for word in ["coordinate", "together", "both agents"]):
        skills.append("coordination")
    if any(word in query.lower() for word in ["organize", "arrange", "setup"]):
        skills.append("planning")

    return skills
