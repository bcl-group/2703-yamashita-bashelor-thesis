#!/bin/bash
# 学内サーバ（GPU）で、C のバッチ 16,384 ＋ 苦手な電流を大幅に増やした比較（jobs_gpu_m3_b16k.txt の 20 本、seed 0〜9）を 4 本ずつ並列に回し、終わったら結果を commit・push する。
# 使い方（リポジトリのルートで）:
#   git pull origin hh-serial
#   nohup bash results/hh_improve/run_server_m3_b16k.sh > results/hh_improve/logs/run_server_m3_b16k.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
echo "start $(date '+%F %T')"
bash results/hh_improve/run_jobs.sh results/hh_improve/jobs_gpu_m3_b16k.txt 4
# ジョブの一覧からタグを取り出して、JSON がないものを知らせる
for t in $(grep -v '^#' results/hh_improve/jobs_gpu_m3_b16k.txt | grep . | sed -E 's/.*--tag ([^ ]+).*/\1/'); do
    [ -f "results/hh_improve/$t.json" ] || echo "FAILED $t（JSON がない。logs/$t.log を見る）"
done
# *.out は .gitignore で除外されているので入れない（明示すると git add ごと失敗する）
git add results/hh_improve/m3_*_b16k_*.json results/hh_improve/m3_*_b16k_*.pt results/hh_improve/logs/m3_*_b16k_*.log
git commit -m "C のバッチ 16,384 ＋ 苦手な電流 4 倍・バッチの半分: m3_hdf_h1k・m3_v2f05d_h4k の seed 0〜9（学内サーバ、自動で commit）"
git pull --rebase origin hh-serial && git push origin hh-serial
echo "all done $(date '+%F %T')"
