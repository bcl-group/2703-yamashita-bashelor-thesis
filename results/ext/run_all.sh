#!/bin/bash
# 上位モデルを追加テスト（f–I、seed 10 本、長時間、分布の外）で採点する。モデルごとに並列
# 使い方（リポジトリのルートで）: bash results/ext/run_all.sh [並列数]
cd "$(dirname "$0")/../.."
mkdir -p results/ext/logs
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
uv run python -X utf8 results/eval_ext.py --list | xargs -P "${1:-4}" -I{} bash -c 'uv run python -u -X utf8 results/eval_ext.py {} > results/ext/logs/{}.log 2>&1; echo "done {}"'
echo all done
