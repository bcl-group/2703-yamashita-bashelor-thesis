#!/bin/bash
# 学内サーバ（GPU）で、用意したジョブを全部順に回す。外出中に放っておけるように 1 本のスクリプトにまとめた
#   1. 第8ラウンド（jobs_gpu_round8.txt、27 本）: 苦手な電流 v1・DAgger・コンダクタンスの項・MLP 3 個
#   2. 第9ラウンド（jobs_gpu_round9.txt、12 本）: 苦手な電流 v2
#   3. SiLU / ReLU の seed 違い（jobs_gpu_mlp.txt、9 本）
#   4. flow function を Γ で採点（score_flow_gamma.py）。1〜3 と並行して回す
# 使い方（リポジトリのルートで）: nohup bash results/hh_improve/run_server_all.sh > results/hh_improve/logs/run_server_all.out 2>&1 &
cd "$(dirname "$0")/../.."
mkdir -p results/hh_improve/logs results/flow_learning
echo "start $(date '+%F %T')"

PY_L4=external/flow-learning-l4dc/.venv/bin/python
if [ -x "$PY_L4" ] && ls external/flow-learning-l4dc/outputs/hhfs/*.pth > /dev/null 2>&1; then
    nohup "$PY_L4" -X utf8 experiments/flow_learning/score_flow_gamma.py external/flow-learning-l4dc/outputs/hhfs/*.pth \
        > results/flow_learning/score_gamma.log 2>&1 &
    echo "flow function の採点を開始（results/flow_learning/score_gamma.log）"
else
    echo "flow function のモデルか環境が見つからないので採点しない"
fi

for J in jobs_gpu_round8.txt jobs_gpu_round9.txt jobs_gpu_mlp.txt; do
    echo "=== $J: $(date '+%F %T') ==="
    bash results/hh_improve/run_jobs.sh "results/hh_improve/$J" 6
done
wait
echo "all done $(date '+%F %T')"
