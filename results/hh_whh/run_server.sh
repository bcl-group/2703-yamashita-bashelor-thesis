#!/bin/bash
# RNN 1 個（hh_mlp_whh.py）のジョブ（jobs_server.txt）を学内サーバの GPU で回し、結果を commit・push する
# GPU のメモリが足りなくならないよう、2 本ずつ並列
# 使い方（リポジトリのルートで）: nohup bash results/hh_whh/run_server.sh > results/hh_whh/logs/run_server.out 2>&1 &
cd "$(dirname "$0")/../.."
mkdir -p results/hh_whh/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
echo "start $(date '+%F %T')"
grep -v '^#' results/hh_whh/jobs_server.txt | grep . | xargs -P "${1:-2}" -I{} bash -c 'args="{}"; tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/"); uv run python -u -X utf8 experiments/hh_surrogate/hh_mlp_whh.py $args > results/hh_whh/logs/$tag.log 2>&1; echo "done $tag: $(grep "^\[" results/hh_whh/logs/$tag.log | tail -1 | cut -c1-120)"'
git add results/hh_whh/*.json results/hh_improve/cft_* results/hh_improve/sp_*
git commit -q -m "RNN 1 個（hh_mlp_whh.py）の閉ループ微調整と seed 違いの結果（学内サーバ GPU、自動で commit）"
git pull -q --rebase && git push -q origin hh-serial
echo "all done $(date '+%F %T')"
