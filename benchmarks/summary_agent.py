"""
Summary Agent for analyzing experience data and generating success/failure summaries.

使用更大规模的模型（如 DeepSeek）来总结分析经验数据，
识别成功和失败的模式，提供改进建议。

Usage:
    agent = SummaryAgent()
    summary = agent.summarize_experiences("locomo")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from src.utils.logger import logger

from benchmarks.experience_store import (
    ExperienceStore,
    ExperienceSummary,
    SampleExperience,
)


# Summary Agent 的 Jinja2 模板环境
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_template_env = Environment(
    loader=FileSystemLoader(_PROMPT_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _get_summary_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
):
    """
    获取用于总结的LLM实例。

    优先级:
    1. 显式传入的参数
    2. conf.yaml 中的 SUMMARY_MODEL 配置
    3. 环境变量 SUMMARY_MODEL__*
    4. 回退到 BASIC_MODEL (使用 DeepSeek 配置)

    Returns:
        BaseChatModel instance
    """
    from src.config import load_yaml_config
    from src.llms.llm import _get_config_file_path

    conf = load_yaml_config(_get_config_file_path())
    summary_conf = conf.get("SUMMARY_MODEL", {})

    # 合并环境变量
    prefix = "SUMMARY_MODEL__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix):].lower()
            summary_conf[conf_key] = value

    # 显式参数覆盖
    if model:
        summary_conf["model"] = model
    if api_key:
        summary_conf["api_key"] = api_key
    if base_url:
        summary_conf["base_url"] = base_url

    summary_conf["temperature"] = temperature

    if not summary_conf.get("model"):
        # 回退: 使用 basic model 配置
        logger.warning(
            "未配置 SUMMARY_MODEL，回退使用 BASIC_MODEL。"
            "建议在 conf.yaml 中配置 SUMMARY_MODEL 使用更大规模的模型（如 DeepSeek）。"
        )
        from src.llms.llm import get_llm_by_type

        return get_llm_by_type("basic")

    # 创建 LLM 实例
    if not summary_conf.get("max_retries"):
        summary_conf["max_retries"] = 3

    # 根据 base_url 判断提供商
    model_base_url = summary_conf.get("base_url", "")

    if "dashscope." in model_base_url:
        from src.llms.providers.dashscope import ChatDashscope

        summary_conf["extra_body"] = {"enable_thinking": False}
        return ChatDashscope(**summary_conf)
    elif "deepseek" in model_base_url or "deepseek" in summary_conf.get("model", "").lower():
        from langchain_deepseek import ChatDeepSeek

        summary_conf["api_base"] = summary_conf.pop("base_url", None)
        return ChatDeepSeek(**summary_conf)
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**summary_conf)


def _select_representative_examples(
    experiences: List[SampleExperience], success: bool, max_examples: int = 5
) -> str:
    """
    选择有代表性的成功/失败案例用于总结。

    策略:
    - 成功案例: 选择成功率最高且有多样性的样本
    - 失败案例: 选择失败率最高且有代表性的样本
    """
    # 按成功率排序
    if success:
        # 优先选择成功率高的
        sorted_exps = sorted(
            experiences, key=lambda e: e.success_rate, reverse=True
        )
    else:
        # 优先选择失败率高的
        sorted_exps = sorted(experiences, key=lambda e: e.success_rate)

    examples = []
    for exp in sorted_exps[:max_examples]:
        runs = exp.success_runs if success else exp.failure_runs
        if not runs:
            continue

        # 取第一个代表性运行
        run = runs[0]

        example = {
            "sample_id": exp.sample_id,
            "query": exp.query[:200],
            "success_rate": f"{exp.success_rate:.1%}",
            "output_preview": (run.output[:300] + "...") if len(run.output) > 300 else run.output,
            "execution_steps": len(run.execution_history),
            "error": run.error[:200] if run.error else None,
        }

        # 提取执行历史中的关键动作序列
        if run.execution_history:
            actions = []
            for entry in run.execution_history:
                if isinstance(entry, dict):
                    action = entry.get("action", "unknown")
                    agent = entry.get("agent_type", "")
                    actions.append(f"{action}({agent})" if agent else action)
            example["action_sequence"] = " → ".join(actions)

        examples.append(example)

    return json.dumps(examples, ensure_ascii=False, indent=2)


class SummaryAgent:
    """
    经验总结Agent

    使用更大规模的模型（推荐DeepSeek）分析经验数据，
    总结成功和失败模式，生成可操作的改进建议。
    """

    def __init__(
        self,
        results_dir: str = "benchmarks/results",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Args:
            results_dir: 经验数据目录
            model: 总结模型名称 (如 "deepseek-chat")
            api_key: 模型API密钥
            base_url: 模型API地址
            temperature: 总结时的temperature (低温更稳定)
        """
        self.store = ExperienceStore(results_dir)
        self.llm = _get_summary_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    def _build_summary_prompt(
        self, benchmark_type: str, experiences: List[SampleExperience]
    ) -> str:
        """构建总结提示词"""
        stats = self.store.get_statistics(benchmark_type)

        total_samples = stats.get("total_samples", 0)
        total_runs = stats.get("total_runs", 0)
        success_rate = stats.get("overall_success_rate", 0) * 100

        # 选择代表性案例
        success_examples = _select_representative_examples(
            experiences, success=True, max_examples=5
        )
        failure_examples = _select_representative_examples(
            experiences, success=False, max_examples=5
        )

        # 统计概览
        statistics_overview = json.dumps(stats, ensure_ascii=False, indent=2)

        # 渲染模板
        template = _template_env.get_template("summary_agent.md")
        prompt = template.render(
            benchmark_type=benchmark_type,
            total_samples=total_samples,
            total_runs=total_runs,
            success_rate=f"{success_rate:.1f}",
            success_examples=success_examples,
            failure_examples=failure_examples,
            statistics_overview=statistics_overview,
        )

        return prompt

    def summarize_experiences(
        self, benchmark_type: str
    ) -> ExperienceSummary:
        """
        对指定基准的经验数据进行总结分析。

        Args:
            benchmark_type: "locomo" or "memgen"

        Returns:
            ExperienceSummary 总结对象
        """
        logger.info(f"开始总结 {benchmark_type} 经验数据...")

        # 加载所有经验数据
        experiences = self.store.load_all_experiences(benchmark_type)
        if not experiences:
            raise ValueError(
                f"未找到 {benchmark_type} 的经验数据，请先运行经验生成。"
            )

        stats = self.store.get_statistics(benchmark_type)
        logger.info(
            f"加载了 {stats['total_samples']} 个样本，"
            f"共 {stats['total_runs']} 次运行"
        )

        # 构建提示词
        prompt = self._build_summary_prompt(benchmark_type, experiences)

        # 调用LLM进行总结
        logger.info("调用Summary LLM进行经验总结...")
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"请对以上 {benchmark_type} 基准测试的经验数据进行全面分析，"
                    f"总结成功经验和失败教训，并提供具体的改进建议。"
                ),
            },
        ]

        response = self.llm.invoke(messages)
        analysis_text = response.content

        # 解析总结结果
        summary = self._parse_summary(benchmark_type, stats, analysis_text)

        # 保存总结
        filepath = self.store.save_summary(summary)
        logger.info(f"经验总结已保存: {filepath}")

        return summary

    def summarize_all(self) -> List[ExperienceSummary]:
        """对所有基准进行总结"""
        summaries = []
        for benchmark_type in ["locomo", "memgen"]:
            try:
                summary = self.summarize_experiences(benchmark_type)
                summaries.append(summary)
            except ValueError as e:
                logger.warning(str(e))
        return summaries

    def _parse_summary(
        self,
        benchmark_type: str,
        stats: Dict[str, Any],
        analysis_text: str,
    ) -> ExperienceSummary:
        """解析LLM输出的总结文本，提取各部分内容"""

        # 定义各段的标题标记
        sections = {
            "success_patterns": [
                "成功经验总结",
                "Success Patterns",
                "1.",
            ],
            "failure_patterns": [
                "失败经验总结",
                "Failure Patterns",
                "2.",
            ],
            "recommendations": [
                "改进建议",
                "Recommendations",
                "4.",
            ],
            "detailed_analysis": [
                "关键差异分析",
                "Success vs Failure",
                "3.",
            ],
        }

        # 简单的段落分割
        parsed = {}
        for key, markers in sections.items():
            parsed[key] = self._extract_section(analysis_text, markers)

        # 如果解析失败，将整个文本作为详细分析
        if not any(parsed.values()):
            parsed["detailed_analysis"] = analysis_text
            parsed["success_patterns"] = ""
            parsed["failure_patterns"] = ""
            parsed["recommendations"] = ""

        return ExperienceSummary(
            benchmark_type=benchmark_type,
            total_samples=stats.get("total_samples", 0),
            total_runs=stats.get("total_runs", 0),
            overall_success_rate=stats.get("overall_success_rate", 0),
            success_patterns=parsed.get("success_patterns", ""),
            failure_patterns=parsed.get("failure_patterns", ""),
            recommendations=parsed.get("recommendations", ""),
            detailed_analysis=parsed.get("detailed_analysis", analysis_text),
        )

    def _extract_section(
        self, text: str, markers: List[str]
    ) -> str:
        """从文本中提取指定段落"""
        lines = text.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            # 检查是否是目标段落的开始
            if any(marker in line for marker in markers) and not in_section:
                in_section = True
                continue

            # 检查是否是下一个段落的开始（以 ### 开头）
            if in_section and line.strip().startswith("###"):
                break

            if in_section:
                section_lines.append(line)

        return "\n".join(section_lines).strip()
