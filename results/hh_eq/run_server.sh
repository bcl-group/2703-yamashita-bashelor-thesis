#!/bin/bash
# 条件を揃えた B・D の学習（jobs_eq.txt）を学内サーバの GPU で回し、終わったら結果を commit して push する
# 使い方（リポジトリのルートで）:
#   git pull && nohup bash results/hh_eq/run_server.sh > results/hh_eq/run_server.out 2>&1 &
# 引数 1: 同時に回す本数（既定 4）
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}   # 使う GPU（tesla4 は 2 枚。GPU=1 bash ... で切り替え）
P=${1:-4}
J=results/hh_eq/jobs_eq.txt
mkdir -p results/hh_eq/logs
echo "start $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
# D の在庫（HH の軌道）を先に 1 回だけ作る（同時に走るジョブが同じ在庫を作り合わないように）
grep -v '^#' "$J" | grep '^rnn3 ' | sed 's/^rnn3 //' | while read -r args; do
    uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args --pools-only
done
grep -v '^#' "$J" | grep . | xargs -P "$P" -I{} bash -c '
    line="{}"; kind=${line%% *}; args=${line#* }
    tag=$(echo "$args" | sed -E "s/.*--tag ([^ ]+).*/\1/")
    if [ "$kind" = rnn3 ]; then
        uv run python -u -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args > results/hh_eq/logs/$tag.log 2>&1
    else
        uv run python -u -X utf8 experiments/hh_surrogate/hh_mlp_whh.py pre $args > results/hh_eq/logs/$tag.log 2>&1
    fi
    echo "done $tag $(date "+%T"): $(grep "^\[" results/hh_eq/logs/$tag.log | tail -1 | cut -c1-120)"'
# *.log は .gitignore で除外されるので、採点の行だけを _score.txt に残す
for f in results/hh_eq/logs/*.log; do t=$(basename "$f" .log); grep "^\[" "$f" > "results/hh_eq/${t}_score.txt"; done
git add results/hh_eq/ results/hh_rnn3/eq_d_*.json results/hh_rnn3/eq_d_*_b[123].pt \
        results/hh_improve/eq_b_*.json results/hh_improve/eq_b_*.pt
git commit -q -m "条件を揃えた B・D の学習（学内サーバ、自動で commit）"
git pull -q --rebase && git push -q origin hh-serial
echo "all done $(date '+%F %T')"
