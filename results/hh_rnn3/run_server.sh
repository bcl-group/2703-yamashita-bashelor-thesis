#!/bin/bash
# RNN 3 個（hh_rnn3.py）の第1・第2ラウンドを学内サーバの GPU で回す（Colab 用の jobs_colab_r1/r2.txt と同じジョブ）
#   1. 在庫（HH の軌道）を 1 本ずつ先に作る（同時に書かないように）
#   2. 第1ラウンド 9 本を 4 本ずつ並列
#   3. 第2ラウンド 12 本を 4 本ずつ並列（一部は第1ラウンドの重みから微調整するので、第1ラウンドの後）
#   4. 結果（.json と .pt）を commit して push
# 使い方（リポジトリのルートで）: nohup bash results/hh_rnn3/run_server.sh > results/hh_rnn3/logs/run_server.out 2>&1 &
cd "$(dirname "$0")/../.."
mkdir -p results/hh_rnn3/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
P=${1:-4}
echo "start $(date '+%F %T')"

uv run python -X utf8 -c "import sys; sys.path.insert(0, 'experiments/hh_surrogate'); import hh_bench as hb; hb.reference(); hb.reference(val=True)"
for J in jobs_colab_r1.txt jobs_colab_r2.txt; do
    grep -v '^#' "results/hh_rnn3/$J" | grep . | while read -r args; do
        uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args --pools-only
    done
done
echo "在庫を作成済み $(date '+%F %T')"

for J in jobs_colab_r1.txt jobs_colab_r2.txt; do
    echo "=== $J: $(date '+%F %T') ==="
    grep -v '^#' "results/hh_rnn3/$J" | grep . | xargs -P "$P" -I{} bash -c 'args="{}"; tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/"); uv run python -u -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args > results/hh_rnn3/logs/$tag.log 2>&1; echo "done $tag: $(grep "^\[" results/hh_rnn3/logs/$tag.log | tail -1 | cut -c1-120)"'
done

git add results/hh_rnn3/*.json results/hh_rnn3/*.pt
git commit -q -m "RNN 3 個（hh_rnn3.py）第1・第2ラウンドの結果（学内サーバ GPU、自動で commit）"
git pull -q --rebase && git push -q origin main
echo "all done $(date '+%F %T')"
