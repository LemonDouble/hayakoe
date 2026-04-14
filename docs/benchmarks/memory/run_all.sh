#!/usr/bin/env bash
# Run all memory benchmark scenarios as isolated subprocesses.
# Output: one JSON line per scenario, appended to results_<timestamp>.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
OUT="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S).jsonl"
: >"$OUT"

for DEVICE in cpu cuda; do
    for SCENARIO in idle1 idle4 gen1 seq4 conc4; do
        echo "==> $DEVICE / $SCENARIO"
        "$PY" "$SCRIPT_DIR/run_one.py" \
            --device "$DEVICE" \
            --scenario "$SCENARIO" \
            2>/dev/null \
            | tee -a "$OUT"
    done
done

echo
echo "Results: $OUT"
