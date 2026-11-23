#!/bin/bash
# RNN 3 個 第3ラウンド（jobs_r3.txt）を、第1・第2ラウンド（run_server.sh）が終わってから学内サーバの GPU で回す
# 使い方（リポジトリのルートで）: nohup bash results/hh_rnn3/run_server_r3.sh > results/hh_rnn3/logs/run_server_r3.out 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
P=${1:-4}
while pgrep -u "$USER" -f 'hh_rnn3/run_server.sh' > /dev/null; do sleep 60; done
echo "start $(date '+%F %T')"
J=results/hh_rnn3/jobs_r3.txt
grep -v '^#' "$J" | grep . | while read -r args; do
    uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args --pools-only
done
grep -v '^#' "$J" | grep . | xargs -P "$P" -I{} bash -c 'args="{}"; tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/"); uv run python -u -X utf8 experiments/hh_surrogate/hh_rnn3.py train $args > results/hh_rnn3/logs/$tag.log 2>&1; echo "done $tag: $(grep "^\[" results/hh_rnn3/logs/$tag.log | tail -1 | cut -c1-120)"'
git add results/hh_rnn3/*.json results/hh_rnn3/*_b[123].pt results/hh_rnn3/jobs_r3.txt results/hh_rnn3/run_server_r3.sh
git commit -q -m "RNN 3 個（hh_rnn3.py）第3ラウンドの結果（学内サーバ GPU、自動で commit）"
git pull -q --rebase && git push -q origin hh-serial
echo "all done $(date '+%F %T')"
