#!/usr/bin/env python3
"""
仅重新生成 summaries，使用已有的 evaluated_results.json 数据。
逻辑与 run_pipeline.py 中 Step 4 + Step 5 完全一致。

用法:
    python scripts/locomo_pipeline/regenerate_summaries.py \
        --run-dir results/locomo/run_20260223_002529 \
        --summary-api-key sk-7374e2abda1141ffa4fe8eb01ae582f7
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.locomo_pipeline.summary_agent import SummaryAgent
from scripts.locomo_pipeline.run_pipeline import build_experience_data

DEFAULT_SUMMARY_BASE_URL = "http://123.57.228.132:8285/api"
DEFAULT_SUMMARY_MODEL = "deepseek-v3.2-20251201-160k-local"


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate summaries from existing evaluated_results.json"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the existing run directory (e.g. results/locomo/run_20260223_002529)",
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=os.getenv("SUMMARY_BASE_URL", DEFAULT_SUMMARY_BASE_URL),
    )
    parser.add_argument(
        "--summary-api-key",
        type=str,
        default=os.getenv("SUMMARY_API_KEY", ""),
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=os.getenv("SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of summaries to generate in parallel (default: 5)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    evaluated_file = os.path.join(run_dir, "evaluated_results.json")

    if not os.path.exists(evaluated_file):
        print(f"❌ 找不到 {evaluated_file}")
        sys.exit(1)

    print("=" * 60)
    print("Regenerate Summaries Only")
    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Summary API:   {args.summary_base_url}")
    print(f"Summary Model: {args.summary_model}")
    print(f"API Key:       {args.summary_api_key[:8]}...{args.summary_api_key[-4:]}" if args.summary_api_key else "API Key: (empty!)")
    print()

    # Step 1: 加载已有的 evaluated_results
    print("[Step 1] Loading evaluated_results.json...")
    with open(evaluated_file, "r", encoding="utf-8") as f:
        evaluated_results = json.load(f)
    print(f"  Loaded {len(evaluated_results)} QA results")

    # Step 2: 生成 summaries（与 run_pipeline.py Step 4 一致）
    print(f"\n[Step 2] Generating experience summaries ({len(evaluated_results)} items)...")
    summary_agent = SummaryAgent(
        base_url=args.summary_base_url,
        api_key=args.summary_api_key,
        model=args.summary_model,
    )
    summaries = summary_agent.summarize_batch(evaluated_results, concurrency=args.concurrency)

    summaries_file = os.path.join(run_dir, "summaries.json")
    with open(summaries_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"  Summaries saved to: {summaries_file}")

    # 统计成功/失败（只有以 "Summary generation failed:" 开头的才是真正的 API 调用失败）
    failed = sum(1 for s in summaries if s.get("summary", "").startswith("Summary generation failed:"))
    print(f"  Summary 成功: {len(summaries) - failed}/{len(summaries)}, 失败: {failed}/{len(summaries)}")

    # Step 3: 重建 experience_data（与 run_pipeline.py Step 5 一致）
    print("\n[Step 3] Rebuilding experience_data.json...")
    experience_data = build_experience_data(evaluated_results, summaries)

    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)
    print(f"  Experience data saved to: {output_file}")

    print("\n" + "=" * 60)
    print("✅ Done! Summaries and experience_data regenerated.")
    print("=" * 60)


if __name__ == "__main__":
    main()

