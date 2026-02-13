"""
CLI entry point for benchmark experience data generation.

Usage:
    # 生成经验数据 (LoCoMo, 每个样本10次, temperature=0.9)
    python -m benchmarks.run_benchmark generate --benchmark locomo --runs 10 --temperature 0.9

    # 生成经验数据 (MemGen)
    python -m benchmarks.run_benchmark generate --benchmark memgen --runs 10

    # 断点续跑
    python -m benchmarks.run_benchmark generate --benchmark locomo --runs 10 --resume

    # 总结经验
    python -m benchmarks.run_benchmark summarize --benchmark locomo

    # 总结所有基准的经验
    python -m benchmarks.run_benchmark summarize --benchmark all

    # 查看统计信息
    python -m benchmarks.run_benchmark stats --benchmark locomo

    # 指定 Summary Agent 的模型
    python -m benchmarks.run_benchmark summarize --benchmark locomo \
        --summary-model deepseek-chat \
        --summary-base-url https://api.deepseek.com/v1 \
        --summary-api-key your-key
"""

import argparse
import asyncio
import json
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger


def cmd_generate(args):
    """运行经验数据生成"""
    from benchmarks.experience_runner import ExperienceRunner

    runner = ExperienceRunner(
        benchmark_type=args.benchmark,
        data_dir=args.data_dir or f"benchmarks/data/{args.benchmark}",
        num_runs=args.runs,
        temperature=args.temperature,
        graph_format=args.graph_format,
        results_dir=args.results_dir,
        max_step_num=args.max_step_num,
        max_plan_iterations=args.max_plan_iterations,
        enable_background_investigation=args.enable_bg_investigation,
        sample_limit=args.sample_limit,
    )

    if args.resume:
        experiences = asyncio.run(runner.run_resume())
    else:
        experiences = asyncio.run(runner.run_all())

    # 输出总结
    print(f"\n{'='*60}")
    print(f"经验数据生成完成")
    print(f"{'='*60}")
    print(f"基准类型: {args.benchmark}")
    print(f"处理样本数: {len(experiences)}")
    print(f"每样本运行数: {args.runs}")
    print(f"Temperature: {args.temperature}")

    total_success = sum(len(e.success_runs) for e in experiences)
    total_runs = sum(e.total_runs for e in experiences)
    overall_rate = total_success / total_runs if total_runs > 0 else 0

    print(f"总运行次数: {total_runs}")
    print(f"成功次数: {total_success}")
    print(f"总体成功率: {overall_rate:.1%}")
    print(f"结果保存目录: {args.results_dir}")


def cmd_summarize(args):
    """运行经验总结"""
    from benchmarks.summary_agent import SummaryAgent

    agent = SummaryAgent(
        results_dir=args.results_dir,
        model=args.summary_model,
        api_key=args.summary_api_key,
        base_url=args.summary_base_url,
        temperature=args.summary_temperature,
    )

    if args.benchmark == "all":
        summaries = agent.summarize_all()
        for summary in summaries:
            _print_summary(summary)
    else:
        summary = agent.summarize_experiences(args.benchmark)
        _print_summary(summary)


def cmd_stats(args):
    """查看统计信息"""
    from benchmarks.experience_store import ExperienceStore

    store = ExperienceStore(args.results_dir)

    if args.benchmark == "all":
        for btype in ["locomo", "memgen"]:
            stats = store.get_statistics(btype)
            _print_stats(btype, stats)
    else:
        stats = store.get_statistics(args.benchmark)
        _print_stats(args.benchmark, stats)


def _print_summary(summary):
    """打印经验总结"""
    print(f"\n{'='*60}")
    print(f"经验总结: {summary.benchmark_type}")
    print(f"{'='*60}")
    print(f"总样本数: {summary.total_samples}")
    print(f"总运行数: {summary.total_runs}")
    print(f"总体成功率: {summary.overall_success_rate:.1%}")

    if summary.success_patterns:
        print(f"\n--- 成功经验 ---")
        print(summary.success_patterns[:500])

    if summary.failure_patterns:
        print(f"\n--- 失败经验 ---")
        print(summary.failure_patterns[:500])

    if summary.recommendations:
        print(f"\n--- 改进建议 ---")
        print(summary.recommendations[:500])

    print(f"\n生成时间: {summary.timestamp}")


def _print_stats(benchmark_type, stats):
    """打印统计信息"""
    print(f"\n--- {benchmark_type} 统计 ---")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Stack-Planner 基准测试经验数据生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 LoCoMo 经验数据 (每样本10次, 高温)
  python -m benchmarks.run_benchmark generate --benchmark locomo --runs 10 --temperature 0.9

  # 生成 MemGen 经验数据 (限制3个样本用于测试)
  python -m benchmarks.run_benchmark generate --benchmark memgen --runs 10 --sample-limit 3

  # 断点续跑
  python -m benchmarks.run_benchmark generate --benchmark locomo --runs 10 --resume

  # 使用 DeepSeek 总结经验
  python -m benchmarks.run_benchmark summarize --benchmark locomo \\
      --summary-model deepseek-chat \\
      --summary-base-url https://api.deepseek.com/v1 \\
      --summary-api-key sk-xxx

  # 查看统计信息
  python -m benchmarks.run_benchmark stats --benchmark all
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # === generate 子命令 ===
    gen_parser = subparsers.add_parser(
        "generate", help="生成经验数据"
    )
    gen_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["locomo", "memgen"],
        help="基准类型",
    )
    gen_parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="每个样本运行次数 (default: 10)",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="LLM temperature (default: 0.9, 高温增加多样性)",
    )
    gen_parser.add_argument(
        "--graph-format",
        type=str,
        default="sp",
        choices=["sp", "sp_xxqg", "base", "xxqg"],
        help="图格式 (default: sp)",
    )
    gen_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录 (default: benchmarks/data/{benchmark})",
    )
    gen_parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="结果保存目录 (default: benchmarks/results)",
    )
    gen_parser.add_argument(
        "--max-step-num",
        type=int,
        default=3,
        help="最大步骤数 (default: 3)",
    )
    gen_parser.add_argument(
        "--max-plan-iterations",
        type=int,
        default=1,
        help="最大规划迭代数 (default: 1)",
    )
    gen_parser.add_argument(
        "--enable-bg-investigation",
        action="store_true",
        default=False,
        help="启用背景调查 (default: False)",
    )
    gen_parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="限制处理的样本数量 (default: 全部)",
    )
    gen_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="断点续跑，跳过已完成的样本",
    )

    # === summarize 子命令 ===
    sum_parser = subparsers.add_parser(
        "summarize", help="总结经验数据"
    )
    sum_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["locomo", "memgen", "all"],
        help="基准类型 (all = 全部)",
    )
    sum_parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="结果目录",
    )
    sum_parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="总结模型名称 (如 deepseek-chat)",
    )
    sum_parser.add_argument(
        "--summary-api-key",
        type=str,
        default=None,
        help="总结模型 API key",
    )
    sum_parser.add_argument(
        "--summary-base-url",
        type=str,
        default=None,
        help="总结模型 API base URL",
    )
    sum_parser.add_argument(
        "--summary-temperature",
        type=float,
        default=0.3,
        help="总结时的 temperature (default: 0.3, 低温更稳定)",
    )

    # === stats 子命令 ===
    stats_parser = subparsers.add_parser(
        "stats", help="查看统计信息"
    )
    stats_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["locomo", "memgen", "all"],
        help="基准类型",
    )
    stats_parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="结果目录",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
