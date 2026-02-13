"""
Benchmark data loaders for LoCoMo and MemGen datasets.

LoCoMo (Long Context Memory): 评估长上下文对话记忆能力的基准
MemGen (Memory Generation): 评估基于记忆生成回复能力的基准

Usage:
    loader = LoCoMoDataLoader("benchmarks/data/locomo")
    samples = loader.load_samples()

    loader = MemGenDataLoader("benchmarks/data/memgen")
    samples = loader.load_samples()
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkSample:
    """单个基准测试样本的统一数据格式"""

    sample_id: str  # 样本唯一标识
    benchmark_type: str  # "locomo" or "memgen"
    conversation_history: List[Dict[str, str]]  # 对话历史
    query: str  # 当前查询/问题
    ground_truth: Optional[str] = None  # 标准答案（如有）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元信息

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark_type": self.benchmark_type,
            "conversation_history": self.conversation_history,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


class BaseBenchmarkDataLoader(ABC):
    """基准数据加载器的抽象基类"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    @abstractmethod
    def load_samples(self) -> List[BenchmarkSample]:
        """加载所有样本数据"""
        pass

    def load_from_json_file(self, filepath: str) -> List[Dict]:
        """通用JSON文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            # 单个样本包装为列表
            return [data]

    def load_from_jsonl_file(self, filepath: str) -> List[Dict]:
        """通用JSONL文件加载"""
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples


class LoCoMoDataLoader(BaseBenchmarkDataLoader):
    """
    LoCoMo 数据加载器

    LoCoMo数据格式预期:
    {
        "conversation_id": "xxx",
        "conversation": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "questions": [
            {
                "question_id": "q1",
                "question": "...",
                "answer": "...",
                "category": "..."
            }
        ]
    }

    也支持扁平格式:
    {
        "id": "xxx",
        "context": "...",
        "question": "...",
        "answer": "..."
    }
    """

    def load_samples(self) -> List[BenchmarkSample]:
        """加载LoCoMo基准数据"""
        all_samples = []
        data_files = self._find_data_files()

        if not data_files:
            raise FileNotFoundError(
                f"在 {self.data_dir} 目录下未找到LoCoMo数据文件。"
                f"请将LoCoMo数据集文件(.json/.jsonl)放置在该目录下。"
            )

        for filepath in data_files:
            raw_data = self._load_file(filepath)
            for item in raw_data:
                samples = self._parse_locomo_item(item)
                all_samples.extend(samples)

        return all_samples

    def _find_data_files(self) -> List[str]:
        """查找目录下所有数据文件"""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith((".json", ".jsonl")):
                files.append(os.path.join(self.data_dir, f))
        return sorted(files)

    def _load_file(self, filepath: str) -> List[Dict]:
        """根据文件扩展名选择加载方式"""
        if filepath.endswith(".jsonl"):
            return self.load_from_jsonl_file(filepath)
        else:
            return self.load_from_json_file(filepath)

    def _parse_locomo_item(self, item: Dict) -> List[BenchmarkSample]:
        """解析单个LoCoMo数据项，可能产生多个样本（一个对话多个问题）"""
        samples = []

        # 格式1: 嵌套格式 (conversation + questions)
        if "conversation" in item and "questions" in item:
            conv_id = item.get("conversation_id", item.get("id", "unknown"))
            conversation = self._normalize_conversation(item["conversation"])

            for q in item["questions"]:
                q_id = q.get("question_id", q.get("id", ""))
                sample_id = f"locomo_{conv_id}_{q_id}"
                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        benchmark_type="locomo",
                        conversation_history=conversation,
                        query=q["question"],
                        ground_truth=q.get("answer"),
                        metadata={
                            "category": q.get("category", ""),
                            "conversation_id": conv_id,
                            "source_file": "",
                        },
                    )
                )

        # 格式2: 扁平格式
        elif "question" in item:
            item_id = item.get("id", item.get("sample_id", "unknown"))
            context = item.get("context", "")
            conversation = (
                self._normalize_conversation(item["conversation"])
                if "conversation" in item
                else [{"role": "system", "content": context}]
                if context
                else []
            )
            samples.append(
                BenchmarkSample(
                    sample_id=f"locomo_{item_id}",
                    benchmark_type="locomo",
                    conversation_history=conversation,
                    query=item["question"],
                    ground_truth=item.get("answer"),
                    metadata={
                        "category": item.get("category", ""),
                    },
                )
            )

        return samples

    def _normalize_conversation(
        self, conversation: List[Dict]
    ) -> List[Dict[str, str]]:
        """统一对话格式"""
        normalized = []
        for msg in conversation:
            role = msg.get("role", msg.get("speaker", "user"))
            content = msg.get("content", msg.get("text", msg.get("utterance", "")))
            normalized.append({"role": role, "content": content})
        return normalized


class MemGenDataLoader(BaseBenchmarkDataLoader):
    """
    MemGen 数据加载器

    MemGen数据格式预期:
    {
        "id": "xxx",
        "memory": [...],  # 或 "context", "history"
        "query": "...",    # 或 "question", "input"
        "response": "..."  # 或 "answer", "output", "ground_truth"
    }
    """

    def load_samples(self) -> List[BenchmarkSample]:
        """加载MemGen基准数据"""
        all_samples = []
        data_files = self._find_data_files()

        if not data_files:
            raise FileNotFoundError(
                f"在 {self.data_dir} 目录下未找到MemGen数据文件。"
                f"请将MemGen数据集文件(.json/.jsonl)放置在该目录下。"
            )

        for filepath in data_files:
            raw_data = self._load_file(filepath)
            for item in raw_data:
                sample = self._parse_memgen_item(item)
                if sample:
                    all_samples.append(sample)

        return all_samples

    def _find_data_files(self) -> List[str]:
        """查找目录下所有数据文件"""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith((".json", ".jsonl")):
                files.append(os.path.join(self.data_dir, f))
        return sorted(files)

    def _load_file(self, filepath: str) -> List[Dict]:
        """根据文件扩展名选择加载方式"""
        if filepath.endswith(".jsonl"):
            return self.load_from_jsonl_file(filepath)
        else:
            return self.load_from_json_file(filepath)

    def _parse_memgen_item(self, item: Dict) -> Optional[BenchmarkSample]:
        """解析单个MemGen数据项"""
        item_id = item.get("id", item.get("sample_id", "unknown"))

        # 提取记忆/上下文
        memory = item.get(
            "memory", item.get("context", item.get("history", []))
        )
        if isinstance(memory, str):
            conversation = [{"role": "system", "content": memory}]
        elif isinstance(memory, list):
            conversation = self._normalize_memory(memory)
        else:
            conversation = []

        # 提取查询
        query = item.get(
            "query", item.get("question", item.get("input", ""))
        )
        if not query:
            return None

        # 提取标准答案
        ground_truth = item.get(
            "response",
            item.get("answer", item.get("output", item.get("ground_truth"))),
        )

        return BenchmarkSample(
            sample_id=f"memgen_{item_id}",
            benchmark_type="memgen",
            conversation_history=conversation,
            query=query,
            ground_truth=ground_truth,
            metadata={
                "category": item.get("category", item.get("type", "")),
            },
        )

    def _normalize_memory(self, memory: List) -> List[Dict[str, str]]:
        """统一记忆格式为对话列表"""
        normalized = []
        for item in memory:
            if isinstance(item, str):
                normalized.append({"role": "system", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", item.get("speaker", "user"))
                content = item.get(
                    "content", item.get("text", item.get("utterance", ""))
                )
                normalized.append({"role": role, "content": content})
        return normalized


def get_data_loader(benchmark_type: str, data_dir: str) -> BaseBenchmarkDataLoader:
    """工厂函数: 根据基准类型返回对应的数据加载器"""
    loaders = {
        "locomo": LoCoMoDataLoader,
        "memgen": MemGenDataLoader,
    }
    if benchmark_type not in loaders:
        raise ValueError(
            f"不支持的基准类型: {benchmark_type}，可用类型: {list(loaders.keys())}"
        )
    return loaders[benchmark_type](data_dir)
