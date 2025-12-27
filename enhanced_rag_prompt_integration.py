# 示例：增强RAG模块以利用enhanced_skills数据


def load_enhanced_skills_data(self):
    """Enhanced version to load skill patterns and quality metrics"""

    skill_files = glob.glob(f"{self._data_dir}/skill_*.json.gz")

    for skill_file in skill_files:
        with gzip.open(skill_file, "rt") as f:
            data = json.load(f)
            skill_category = data["metadata"]["skill_category"]

            for episode in data["episodes"]:
                # 使用enhanced_skills而不是原始skills
                if "enhanced_skills" in episode:
                    for agent_id_str, enhanced_data in episode[
                        "enhanced_skills"
                    ].items():
                        agent_id = int(agent_id_str)

                        # 提取技能模式
                        skill_patterns = enhanced_data.get("skill_patterns", [])
                        decision_points = enhanced_data.get("decision_points", [])
                        coordination_req = enhanced_data.get(
                            "coordination_requirements", {}
                        )
                        action_efficiency = enhanced_data.get("action_efficiency", {})

                        # 构建富文本描述
                        enhanced_trace = self._build_enhanced_trace(
                            episode["instruction"],
                            enhanced_data["enhanced_skill_description"],
                            skill_patterns,
                            decision_points,
                            coordination_req,
                            action_efficiency,
                        )

                        info = {
                            "instruction": episode["instruction"],
                            "skill_category": skill_category,
                            "task_type": episode.get("task_type", "Unknown"),
                            "complexity": episode.get("complexity", "Unknown"),
                            "agent_id": agent_id,
                            "trace": enhanced_trace,  # 使用增强的trace
                            "skill_patterns": [p["skill_name"] for p in skill_patterns],
                            "decision_count": len(decision_points),
                            "coordination_required": coordination_req.get(
                                "requires_coordination", False
                            ),
                            "efficiency_score": action_efficiency.get(
                                "efficiency_score", 0
                            ),
                            "quality_score": episode.get(
                                "episode_quality_metrics", {}
                            ).get("overall_quality", 0),
                            "file": skill_file,
                            "episode_id": episode["episode_id"],
                        }

                        self.data_dict[self.index] = info
                        self.index += 1


def _build_enhanced_trace(
    self, instruction, skill_desc, patterns, decisions, coordination, efficiency
):
    """构建包含技能分析的增强trace"""

    trace = f"Task: {instruction}\n\n"
    trace += f"Skill Analysis: {skill_desc}\n\n"

    if patterns:
        trace += "Key Skill Patterns:\n"
        for pattern in patterns:
            trace += f"- {pattern['skill_name']}: {pattern['description']}\n"
        trace += "\n"

    if decisions:
        trace += f"Decision Points: {len(decisions)} critical decisions made\n"
        for i, decision in enumerate(decisions[:3]):  # 显示前3个
            trace += f"  {i+1}. {decision.get('description', 'Strategic decision')}\n"
        trace += "\n"

    if coordination.get("requires_coordination"):
        trace += f"Coordination: Required multi-agent coordination ({coordination.get('coordination_actions_count', 0)} coordination actions)\n\n"

    if efficiency.get("efficiency_score"):
        trace += f"Efficiency: {efficiency['efficiency_score']:.2f} success rate ({efficiency.get('metrics', {}).get('successful_steps', 0)}/{efficiency.get('metrics', {}).get('total_steps', 1)} actions)\n\n"

    return trace


def retrieve_top_k_with_filters(
    self,
    query: str,
    top_k: int = 1,
    agent_id: int = 0,
    skill_filter=None,
    min_quality=0.0,
    coordination_required=None,
):
    """增强的检索，支持技能模式和质量过滤"""

    # 基础检索
    query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

    # 应用过滤器
    filtered_indices = []
    for index in self.data_dict:
        episode = self.data_dict[index]

        # Agent ID过滤
        if episode.get("agent_id") != agent_id:
            continue

        # 技能模式过滤
        if skill_filter and not any(
            skill in episode.get("skill_patterns", []) for skill in skill_filter
        ):
            continue

        # 质量过滤
        if episode.get("quality_score", 0) < min_quality:
            continue

        # 协作需求过滤
        if (
            coordination_required is not None
            and episode.get("coordination_required") != coordination_required
        ):
            continue

        filtered_indices.append(index)

    if not filtered_indices:
        # 降级到基础检索
        return self.retrieve_top_k_given_query(query, top_k, agent_id)

    # 在过滤后的结果中计算相似度
    embeddings = torch.stack(
        [self.data_dict[idx]["embedding"] for idx in filtered_indices]
    )
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=min(top_k, len(filtered_indices)))

    # 转换回原始索引
    original_indices = [filtered_indices[i] for i in indices.cpu().numpy()]

    return scores.cpu().numpy(), np.array(original_indices)
