from enum import Enum


class SubAgentType(Enum):
    """子Agent类型枚举，定义可委派的专项Agent"""

    RESEARCHER = "researcher"  # 负责信息检索与研究
    CODER = "coder"  # 负责代码生成与执行
    REPORTER = "reporter"  # 负责结果整理与报告生成
    PLANNER = "replanner"  # 负责复杂任务分解和规划


from src.graph.sp_nodes import (
    researcher_node,
    # coder_node,  # 已禁用 coder 子代理
    reporter_node,
    researcher_xxqg_node,
    reporter_xxqg_node,
    sp_planner_node,
)

# 定义可用的子Agent列表，绑定名称与节点函数
sub_agents_sp = [
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
        "node": researcher_node,
    },
    # {  # 已禁用 coder 子代理
    #     "name": SubAgentType.CODER.value,
    #     "description": "Code generation and execution for math or code problems",
    #     "node": coder_node,
    # },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
        "node": reporter_node,
    },
]


sub_agents_sp_xxqg = [
    {
        "name": SubAgentType.PLANNER.value,
        # 将问题拆解成方便处理的子任务，来更好的指导任务规划
        "description": "Decompose search problems into manageable subtasks to better guide research step. Don't contain any requirements about report writing in task description, this agent can only handle **search steps planning**. You MUST and Only use it at the beginning of the task.",
        "node": sp_planner_node,
    },
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
        "node": researcher_xxqg_node,
    },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
        "node": reporter_xxqg_node,
    },
]


sub_agents_locomo = [
    # NOTE: Reporter agent 已禁用 for LoCoMo.
    # 原因：Reporter 的 prompt 模板面向网络搜索报告生成，不适合对话 QA 任务。
    # 启用后 reporter 会生成完全无关的内容（如网页爬取结果）。
    # 对 LoCoMo 场景，CentralAgent 自身的 THINK 推理即可得出答案，
    # _handle_finish 会直接使用 decision.reasoning 作为最终报告。
    # 如需恢复，取消注释以下代码：
    # {
    #     "name": SubAgentType.REPORTER.value,
    #     "description": "Result organization and answer generation based on conversation context",
    #     "node": reporter_node,
    # },
    # NOTE: Researcher agent designed but commented out for LoCoMo.
    # Uncomment for benchmarks that need web search / retrieval.
    # {
    #     "name": SubAgentType.RESEARCHER.value,
    #     "description": "Information collection and research from external sources",
    #     "node": researcher_node,
    # },
]


def get_sub_agents_by_global_type(graph_type: str):
    """
    根据图类型返回可用的子Agent列表
    Args:
        graph_type (str): 图类型，例如 "sp", "sp_xxqg", 或 "locomo"
    Returns:
        List[Dict]: 包含子Agent名称、节点和描述的列表
    """
    if graph_type == "sp" or graph_type == "base":
        return sub_agents_sp
    elif graph_type == "sp_xxqg":
        return sub_agents_sp_xxqg
    elif graph_type == "locomo":
        return sub_agents_locomo
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
