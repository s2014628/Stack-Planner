#!/usr/bin/env bash
# Run QA bench pipeline commands in the background with nohup.
#
# Usage (from repo root):
#   chmod +x scripts/qa_bench_pipeline/run_nohup.sh   # once
#
#   # Continue / run GSM8K benchmark (same as: python -m scripts.qa_bench_pipeline.run_benchmark ...)
#   ./scripts/qa_bench_pipeline/run_nohup.sh benchmark \
#       --datasets gsm8k --split train --max-samples 100 --num-runs 3 \
#       --output-dir results/qa_bench/run_20260311_062350/gsm8k \
#       --concurrency 3 --run-concurrency 5
#
#   # Full pipeline (new timestamped run_* dir under --output-dir)
#   ./scripts/qa_bench_pipeline/run_nohup.sh pipeline \
#       --datasets gsm8k --max-samples 50 --num-runs 3
#
#   # Post-process existing run
#   ./scripts/qa_bench_pipeline/run_nohup.sh experience \
#       --run-dir results/qa_bench/run_20260311_062350 --per-run --per-run-summary
#
# Environment:
#   PYTHON   Python executable (default: python3)
#   LOG_DIR  Log directory (default: results/qa_bench_logs)
#   LOG_FILE Explicit log path (optional; overrides auto naming)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
shift

PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/results/qa_bench_logs}"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${LOG_FILE:-}" ]]; then
  LOG_PATH="$LOG_FILE"
else
  LOG_PATH="$LOG_DIR/qa_bench_${MODE}_${TS}.log"
fi

case "$MODE" in
  benchmark)
    CMD=("$PYTHON" -m scripts.qa_bench_pipeline.run_benchmark "$@")
    ;;
  pipeline)
    CMD=("$PYTHON" -m scripts.qa_bench_pipeline.run_pipeline "$@")
    ;;
  experience)
    CMD=("$PYTHON" -m scripts.qa_bench_pipeline.generate_experience_from_run "$@")
    ;;
  *)
    echo "Unknown mode: $MODE (use: benchmark | pipeline | experience)" >&2
    exit 1
    ;;
esac

export PYTHONUNBUFFERED=1

nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
PID=$!

echo "Started in background: PID=$PID"
echo "  Log: $LOG_PATH"
echo "  Tail: tail -f $LOG_PATH"

PID_FILE="${LOG_PATH%.log}.pid"
echo "$PID" >"$PID_FILE"
echo "  PID file: $PID_FILE"
