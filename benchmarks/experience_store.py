"""
Experience data store for managing benchmark run results.

负责存储、加载和管理经验数据，包括:
- 单次运行结果 (RunResult)
- 样本的多次运行结果集合 (SampleExperience)
- 按success/failure分类的经验汇总
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunResult:
    """单次运行结果"""

    run_id: int  # 第几次运行 (0-indexed)
    sample_id: str  # 对应的样本ID
    success: bool  # 是否成功
    output: str  # 模型输出
    ground_truth: Optional[str]  # 标准答案
    execution_history: List[Dict[str, Any]]  # 执行历史 (来自memory_stack)
    statistics: Dict[str, Any]  # 运行统计 (token数, 耗时等)
    error: Optional[str] = None  # 如果失败，记录错误信息
    timestamp: str = ""  # 运行时间

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "success": self.success,
            "output": self.output,
            "ground_truth": self.ground_truth,
            "execution_history": self.execution_history,
            "statistics": self.statistics,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        return cls(**data)


@dataclass
class SampleExperience:
    """单个样本的多次运行经验集合"""

    sample_id: str
    benchmark_type: str  # "locomo" or "memgen"
    query: str  # 原始查询
    conversation_context: str  # 对话上下文摘要
    runs: List[RunResult] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def success_runs(self) -> List[RunResult]:
        return [r for r in self.runs if r.success]

    @property
    def failure_runs(self) -> List[RunResult]:
        return [r for r in self.runs if not r.success]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return len(self.success_runs) / len(self.runs)

    def add_run(self, result: RunResult):
        self.runs.append(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark_type": self.benchmark_type,
            "query": self.query,
            "conversation_context": self.conversation_context,
            "total_runs": self.total_runs,
            "success_rate": self.success_rate,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleExperience":
        runs = [RunResult.from_dict(r) for r in data.get("runs", [])]
        return cls(
            sample_id=data["sample_id"],
            benchmark_type=data["benchmark_type"],
            query=data["query"],
            conversation_context=data.get("conversation_context", ""),
            runs=runs,
        )


@dataclass
class ExperienceSummary:
    """经验总结数据结构"""

    benchmark_type: str
    total_samples: int
    total_runs: int
    overall_success_rate: float
    success_patterns: str  # 成功经验总结
    failure_patterns: str  # 失败经验总结
    recommendations: str  # 改进建议
    detailed_analysis: str  # 详细分析
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_type": self.benchmark_type,
            "total_samples": self.total_samples,
            "total_runs": self.total_runs,
            "overall_success_rate": self.overall_success_rate,
            "success_patterns": self.success_patterns,
            "failure_patterns": self.failure_patterns,
            "recommendations": self.recommendations,
            "detailed_analysis": self.detailed_analysis,
            "timestamp": self.timestamp,
        }


class ExperienceStore:
    """经验数据存储管理器"""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_sample_experience(self, experience: SampleExperience) -> str:
        """保存单个样本的经验数据，返回文件路径"""
        sample_dir = os.path.join(
            self.results_dir, experience.benchmark_type, experience.sample_id
        )
        os.makedirs(sample_dir, exist_ok=True)

        filepath = os.path.join(sample_dir, "experience.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experience.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def save_run_result(
        self, benchmark_type: str, sample_id: str, result: RunResult
    ) -> str:
        """增量保存单次运行结果"""
        sample_dir = os.path.join(self.results_dir, benchmark_type, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        filepath = os.path.join(sample_dir, f"run_{result.run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def load_sample_experience(
        self, benchmark_type: str, sample_id: str
    ) -> Optional[SampleExperience]:
        """加载单个样本的经验数据"""
        filepath = os.path.join(
            self.results_dir, benchmark_type, sample_id, "experience.json"
        )
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SampleExperience.from_dict(data)

    def load_all_experiences(
        self, benchmark_type: str
    ) -> List[SampleExperience]:
        """加载指定基准的所有经验数据"""
        benchmark_dir = os.path.join(self.results_dir, benchmark_type)
        if not os.path.exists(benchmark_dir):
            return []

        experiences = []
        for sample_id in sorted(os.listdir(benchmark_dir)):
            sample_dir = os.path.join(benchmark_dir, sample_id)
            if not os.path.isdir(sample_dir):
                continue
            exp = self.load_sample_experience(benchmark_type, sample_id)
            if exp:
                experiences.append(exp)

        return experiences

    def save_summary(self, summary: ExperienceSummary) -> str:
        """保存经验总结"""
        summary_dir = os.path.join(self.results_dir, "summaries")
        os.makedirs(summary_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            summary_dir,
            f"summary_{summary.benchmark_type}_{timestamp}.json",
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def get_statistics(self, benchmark_type: str) -> Dict[str, Any]:
        """获取指定基准的统计信息"""
        experiences = self.load_all_experiences(benchmark_type)

        if not experiences:
            return {"message": f"暂无 {benchmark_type} 的经验数据"}

        total_samples = len(experiences)
        total_runs = sum(e.total_runs for e in experiences)
        total_successes = sum(len(e.success_runs) for e in experiences)
        overall_success_rate = total_successes / total_runs if total_runs > 0 else 0.0

        # 按成功率分组
        high_success = [e for e in experiences if e.success_rate >= 0.7]
        medium_success = [
            e for e in experiences if 0.3 <= e.success_rate < 0.7
        ]
        low_success = [e for e in experiences if e.success_rate < 0.3]

        return {
            "benchmark_type": benchmark_type,
            "total_samples": total_samples,
            "total_runs": total_runs,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
            "high_success_samples": len(high_success),
            "medium_success_samples": len(medium_success),
            "low_success_samples": len(low_success),
        }
