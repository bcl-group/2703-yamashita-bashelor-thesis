#!/bin/bash
# 使い方: bash results/hh_serial/run_jobs.sh jobs_1_blocks.txt 並列数
# jobs ファイルの 1 行 = hh_serial.py の引数 1 回分。ログは results/hh_serial/logs/<tag>.log
cd "$(dirname "$0")/../.."
mkdir -p results/hh_serial/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
grep -v '^#' "results/hh_serial/$(basename "$1")" | grep . | xargs -P "${2:-4}" -I{} bash -c 'args="{}"; tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/"); uv run python -u -X utf8 experiments/hh_surrogate/hh_serial.py $args > results/hh_serial/logs/$tag.log 2>&1; echo "done $tag: $(grep "^\[" results/hh_serial/logs/$tag.log | tail -1)"'
