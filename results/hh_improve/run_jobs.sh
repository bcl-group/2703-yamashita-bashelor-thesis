#!/bin/bash
# 使い方: bash results/hh_improve/run_jobs.sh jobs.txt 並列数
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
grep -v '^#' "$1" | grep . | xargs -P "${2:-3}" -I{} bash -c 'args="{}"; tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/"); uv run python -u -X utf8 experiments/hh_surrogate/hh_improve.py $args > results/hh_improve/logs/$tag.log 2>&1; echo "done $tag: $(grep "^\[" results/hh_improve/logs/$tag.log | tail -1)"'
