"""
Experience data generation runner.

对 LoCoMo 和 MemGen 基准数据集中的每个样本运行多次,
收集成功/失败经验数据用于后续总结分析。

Usage:
    runner = ExperienceRunner(
        benchmark_type="locomo",
        data_dir="benchmarks/data/locomo",
        num_runs=10,
        temperature=0.9,
    )
    await runner.run_all()
"""

import asyncio
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger import logger
from src.utils.statistics import GlobalStatistics

from benchmarks.data_loader import BenchmarkSample, get_data_loader
from benchmarks.experience_store import (
    ExperienceStore,
    RunResult,
    SampleExperience,
)


def _build_benchmark_query(sample: BenchmarkSample) -> str:
    """
    将基准样本转换为 Stack-Planner 可处理的用户查询。

    将对话历史和查询问题拼接为完整的输入。
    """
    parts = []

    # 添加对话上下文
    if sample.conversation_history:
        parts.append("## 对话上下文 (Conversation Context)\n")
        for msg in sample.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        parts.append("")

    # 添加当前问题
    parts.append("## 当前问题 (Current Question)\n")
    parts.append(sample.query)

    # 如有元信息中的类别，添加提示
    category = sample.metadata.get("category", "")
    if category:
        parts.append(f"\n[问题类别: {category}]")

    return "\n".join(parts)


def _evaluate_result(
    output: str, ground_truth: Optional[str], sample: BenchmarkSample
) -> bool:
    """
    评估运行结果是否成功。

    策略:
    1. 如果有 ground_truth，检查输出中是否包含关键答案信息
    2. 如果没有 ground_truth，检查输出是否合理 (非空、非错误)
    """
    if not output or output.strip() == "":
        return False

    # 检查是否包含错误信息
    error_indicators = [
        "报告生成失败",
        "执行失败",
        "错误",
        "Error",
        "Failed",
        "任务失败",
    ]
    for indicator in error_indicators:
        if indicator in output and len(output) < 200:
            return False

    # 如果有标准答案，进行简单的包含性检查
    if ground_truth:
        gt_lower = ground_truth.lower().strip()
        output_lower = output.lower().strip()

        # 对于短答案，检查是否包含在输出中
        if len(gt_lower) < 100:
            return gt_lower in output_lower

        # 对于长答案，检查关键词重叠率
        gt_words = set(gt_lower.split())
        output_words = set(output_lower.split())
        if gt_words:
            overlap = len(gt_words & output_words) / len(gt_words)
            return overlap >= 0.3

    # 没有标准答案时，认为有实质输出即为成功
    return len(output.strip()) > 50


class ExperienceRunner:
    """经验数据生成运行器"""

    def __init__(
        self,
        benchmark_type: str,
        data_dir: str,
        num_runs: int = 10,
        temperature: float = 0.9,
        graph_format: str = "sp",
        results_dir: str = "benchmarks/results",
        max_step_num: int = 3,
        max_plan_iterations: int = 1,
        enable_background_investigation: bool = False,
        sample_limit: Optional[int] = None,
    ):
        """
        Args:
            benchmark_type: "locomo" or "memgen"
            data_dir: 基准数据目录
            num_runs: 每个样本运行次数
            temperature: LLM temperature (高温增加多样性)
            graph_format: 使用的图格式 ("sp", "sp_xxqg", etc.)
            results_dir: 结果保存目录
            max_step_num: 最大步骤数
            max_plan_iterations: 最大规划迭代数
            enable_background_investigation: 是否启用背景调查
            sample_limit: 限制处理的样本数量 (None=全部)
        """
        self.benchmark_type = benchmark_type
        self.data_dir = data_dir
        self.num_runs = num_runs
        self.temperature = temperature
        self.graph_format = graph_format
        self.max_step_num = max_step_num
        self.max_plan_iterations = max_plan_iterations
        self.enable_background_investigation = enable_background_investigation
        self.sample_limit = sample_limit

        self.store = ExperienceStore(results_dir)
        self.data_loader = get_data_loader(benchmark_type, data_dir)

    def _set_temperature(self):
        """通过环境变量设置高温参数"""
        os.environ["BASIC_MODEL__temperature"] = str(self.temperature)
        logger.info(f"设置 temperature={self.temperature}")

    def _reset_temperature(self):
        """重置温度设置"""
        if "BASIC_MODEL__temperature" in os.environ:
            del os.environ["BASIC_MODEL__temperature"]

    async def _run_single(
        self, sample: BenchmarkSample, run_id: int
    ) -> RunResult:
        """执行单次运行"""
        from src.llms.llm import _llm_cache

        # 清除LLM缓存，确保新的temperature生效
        _llm_cache.clear()

        # 重置统计
        from src.utils.statistics import global_statistics

        global_statistics.reset()

        query = _build_benchmark_query(sample)
        logger.info(
            f"[{sample.sample_id}] 运行 #{run_id + 1}/{self.num_runs}"
        )

        output = ""
        execution_history = []
        statistics = {}
        error = None
        success = False

        try:
            from src.graph.sp_nodes import init_agents

            init_agents(self.graph_format)

            # 获取对应的图
            from src.graph.builder import get_graph_by_format

            graph = get_graph_by_format(self.graph_format, with_memory=False)

            initial_state = {
                "messages": [{"role": "user", "content": query}],
                "auto_accepted_plan": True,
                "enable_background_investigation": self.enable_background_investigation,
                "user_query": query,
            }

            config = {
                "configurable": {
                    "thread_id": f"bench_{sample.sample_id}_run{run_id}",
                    "max_plan_iterations": self.max_plan_iterations,
                    "max_step_num": self.max_step_num,
                },
                "recursion_limit": 100,
            }

            # 执行工作流
            final_state = None
            async for s in graph.astream(
                input=initial_state, config=config, stream_mode="values"
            ):
                if isinstance(s, dict):
                    final_state = s

            if final_state:
                output = final_state.get("final_report", "")
                memory_stack_raw = final_state.get("memory_stack", "[]")
                if isinstance(memory_stack_raw, str):
                    try:
                        execution_history = json.loads(memory_stack_raw)
                    except json.JSONDecodeError:
                        execution_history = []
                elif isinstance(memory_stack_raw, list):
                    execution_history = memory_stack_raw

            statistics = global_statistics.get_statistics()
            success = _evaluate_result(output, sample.ground_truth, sample)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(
                f"[{sample.sample_id}] 运行 #{run_id + 1} 失败: {error}"
            )

        result = RunResult(
            run_id=run_id,
            sample_id=sample.sample_id,
            success=success,
            output=output,
            ground_truth=sample.ground_truth,
            execution_history=execution_history,
            statistics=statistics,
            error=error,
        )

        # 增量保存
        self.store.save_run_result(
            self.benchmark_type, sample.sample_id, result
        )
        logger.info(
            f"[{sample.sample_id}] 运行 #{run_id + 1} 完成: "
            f"{'成功' if success else '失败'}"
        )

        return result

    async def run_sample(self, sample: BenchmarkSample) -> SampleExperience:
        """对单个样本运行多次，收集经验数据"""
        logger.info(
            f"开始处理样本 {sample.sample_id}，共 {self.num_runs} 次运行"
        )

        # 构建对话上下文摘要
        context_summary = ""
        if sample.conversation_history:
            ctx_parts = []
            for msg in sample.conversation_history[:5]:  # 最多取前5条
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                ctx_parts.append(f"{msg.get('role', 'user')}: {content}")
            context_summary = "\n".join(ctx_parts)
            if len(sample.conversation_history) > 5:
                context_summary += f"\n... (共 {len(sample.conversation_history)} 条消息)"

        experience = SampleExperience(
            sample_id=sample.sample_id,
            benchmark_type=self.benchmark_type,
            query=sample.query,
            conversation_context=context_summary,
        )

        for run_id in range(self.num_runs):
            result = await self._run_single(sample, run_id)
            experience.add_run(result)

            # 每次运行后保存整体经验
            self.store.save_sample_experience(experience)

        logger.info(
            f"样本 {sample.sample_id} 完成: "
            f"成功率 {experience.success_rate:.1%} "
            f"({len(experience.success_runs)}/{experience.total_runs})"
        )

        return experience

    async def run_all(self) -> List[SampleExperience]:
        """运行所有样本"""
        logger.info(
            f"开始经验数据生成: benchmark={self.benchmark_type}, "
            f"runs_per_sample={self.num_runs}, "
            f"temperature={self.temperature}"
        )

        # 设置高温
        self._set_temperature()

        try:
            # 加载样本
            samples = self.data_loader.load_samples()
            if self.sample_limit:
                samples = samples[: self.sample_limit]

            logger.info(f"共加载 {len(samples)} 个样本")

            all_experiences = []
            for i, sample in enumerate(samples):
                logger.info(
                    f"进度: {i + 1}/{len(samples)} - {sample.sample_id}"
                )
                experience = await self.run_sample(sample)
                all_experiences.append(experience)

            # 输出总体统计
            stats = self.store.get_statistics(self.benchmark_type)
            logger.info(f"经验数据生成完成，统计: {json.dumps(stats, ensure_ascii=False)}")

            return all_experiences

        finally:
            self._reset_temperature()

    async def run_resume(
        self, skip_completed: bool = True
    ) -> List[SampleExperience]:
        """
        断点续跑: 跳过已完成的样本。

        Args:
            skip_completed: 是否跳过已完成指定次数运行的样本
        """
        logger.info("断点续跑模式启动")
        self._set_temperature()

        try:
            samples = self.data_loader.load_samples()
            if self.sample_limit:
                samples = samples[: self.sample_limit]

            all_experiences = []
            skipped = 0

            for i, sample in enumerate(samples):
                # 检查已有结果
                existing = self.store.load_sample_experience(
                    self.benchmark_type, sample.sample_id
                )

                if (
                    skip_completed
                    and existing
                    and existing.total_runs >= self.num_runs
                ):
                    logger.info(
                        f"跳过已完成样本: {sample.sample_id} "
                        f"(已运行 {existing.total_runs} 次)"
                    )
                    all_experiences.append(existing)
                    skipped += 1
                    continue

                logger.info(
                    f"进度: {i + 1}/{len(samples)} - {sample.sample_id}"
                )
                experience = await self.run_sample(sample)
                all_experiences.append(experience)

            logger.info(
                f"断点续跑完成: 跳过 {skipped} 个已完成样本"
            )
            return all_experiences

        finally:
            self._reset_temperature()
