#!/bin/bash
# 全モデルを Γ で採点し直す（6 本ずつ並列）。使い方（リポジトリのルートで）: bash results/gamma/run_all.sh
cd "$(dirname "$0")/../.."
mkdir -p results/gamma/logs
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
uv run python -X utf8 results/rescore_gamma.py --list | xargs -P 6 -I{} bash -c 'uv run python -u -X utf8 results/rescore_gamma.py {} > results/gamma/logs/{}.log 2>&1; echo "done {}: $(grep "^\[" results/gamma/logs/{}.log | tail -1 | cut -c1-90)"'
echo all done
