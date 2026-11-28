#!/bin/bash
# 条件を揃えた B・D の学習（jobs_eq.txt）を学内サーバの GPU で回し、終わったら結果を commit して push する
# 使い方（リポジトリのルートで）:
#   git pull && nohup bash results/hh_eq/run_server.sh > results/hh_eq/run_server.out 2>&1 &
# 引数 1: 同時に回す本数（既定 4）。GPU=1 を付けると 2 枚目の GPU を使う
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=${1:-4}
J=results/hh_eq/jobs_eq.txt
mkdir -p results/hh_eq/logs
echo "start $(date '+%F %T')  GPU $CUDA_VISIBLE_DEVICES  並列 $P"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
# D の在庫（HH の軌道）を先に 1 回だけ作る（同時に走るジョブが同じ在庫を作り合わないように）
grep -v '^#' "$J" | grep '^rnn3 ' | sed 's/^rnn3 //' | while read -r args; do
    uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args --pools-only || echo "FAILED 在庫: $args"
done
grep -v '^#' "$J" | grep . | xargs -P "$P" -L 1 bash results/hh_eq/run_one.sh
# *.log は .gitignore で除外されるので、採点の行だけを _score.txt に残す
for f in results/hh_eq/logs/*.log; do t=$(basename "$f" .log); grep '^\[' "$f" > "results/hh_eq/${t}_score.txt"; done
files=$(ls results/hh_eq/*.txt results/hh_eq/*.md results/hh_eq/*.sh \
           results/hh_rnn3/eq_d_*.json results/hh_rnn3/eq_d_*_b[123].pt \
           results/hh_improve/eq_b_*.json results/hh_improve/eq_b_*.pt \
           results/hh_improve/eq_b0_*.json results/hh_improve/eq_b0_*.pt 2>/dev/null)
echo "commit するファイル: $(echo $files | wc -w) 個"
git add $files || { echo "FAILED git add"; exit 1; }
git commit -q -m "条件を揃えた B・D の学習（学内サーバ、自動で commit）" || { echo "FAILED git commit"; exit 1; }
git pull -q --rebase --autostash || { echo "FAILED git pull（手で直して push）"; exit 1; }
git push -q origin main || { echo "FAILED git push"; exit 1; }
echo "all done $(date '+%F %T')"
