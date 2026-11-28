#!/bin/bash
# 学内サーバ（GPU）で、ジョブ一覧（1 行 = train.py の引数 1 本分、# で始まる行は飛ばす）を P 本ずつ並列に学習し、
# 終わったら結果を commit・push する。
# 使い方（リポジトリのルートで）:
#   git pull origin main
#   nohup bash experiments/mlp3/run_server.sh experiments/mlp3/jobs/m3_b16k.txt 4 > results/mlp3/run_server.out 2>&1 &
#
# 結果: results/mlp3/<tag>/（train.py の先頭を参照）と、画面の出力 results/mlp3/logs/<tag>.log
# git に入れるもの: 重み・損失・採点・図・ログ。応答の数値 responses_*.npz（1 本 約 75 MB）は .gitignore で除外し、
#   サーバに残す（重みから evaluate.py で同じものを作り直せる）
set -u
cd "$(dirname "$0")/../.."
JOBS=$1
P=${2:-4}
mkdir -p results/mlp3/logs
export PYTHONUNBUFFERED=1
echo "start $(date '+%F %T')  $JOBS（$P 本ずつ並列）"
tags=$(grep -v '^#' "$JOBS" | grep . | sed -E 's/.*--tag ([^ ]+).*/\1/')

grep -v '^#' "$JOBS" | grep . | xargs -P "$P" -I{} bash -c '
    args="{}"
    tag=$(echo $args | sed -E "s/.*--tag ([^ ]+).*/\1/")
    uv run python -u -X utf8 experiments/mlp3/train.py $args --device cuda > results/mlp3/logs/$tag.log 2>&1
    echo "done $tag: $(grep "(best) test" results/mlp3/logs/$tag.log | tail -1 | sed -E "s/不一致.*//")"'

done_dirs=()
for t in $tags; do
    if [ -f "results/mlp3/$t/metrics.json" ]; then
        done_dirs+=("results/mlp3/$t" "results/mlp3/logs/$t.log")
    else
        echo "FAILED $t（results/mlp3/logs/$t.log を見る）"
    fi
done
if [ ${#done_dirs[@]} -gt 0 ]; then
    git add "${done_dirs[@]}"
    git commit -m "MLP 3 個の学習 $(basename "$JOBS" .txt)（学内サーバ、自動で commit）"
    git pull --rebase origin main && git push origin main
fi
echo "all done $(date '+%F %T')"
