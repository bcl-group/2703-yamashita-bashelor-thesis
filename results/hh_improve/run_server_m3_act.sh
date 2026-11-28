#!/bin/bash
# 学内サーバ（GPU）で、C の活性化関数の比較（jobs_gpu_m3_act.txt の 4 本）を並列に回し、終わったら結果を commit・push する。
# 使い方（リポジトリのルートで）:
#   git pull origin main
#   nohup bash results/hh_improve/run_server_m3_act.sh > results/hh_improve/logs/run_server_m3_act.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
echo "start $(date '+%F %T')"
bash results/hh_improve/run_jobs.sh results/hh_improve/jobs_gpu_m3_act.txt 4
for t in m3_hdf_relu_s1 m3_hdf_relu_s2 m3_hdf_s1 m3_hdf_s2; do
    [ -f "results/hh_improve/$t.json" ] || echo "FAILED $t（JSON がない。logs/$t.log を見る）"
done
# *.out は .gitignore で除外されているので入れない（明示すると git add ごと失敗する）
git add results/hh_improve/m3_hdf_*.json results/hh_improve/m3_hdf_*.pt results/hh_improve/logs/m3_hdf_*.log
git commit -m "C の活性化関数の比較: ReLU・SiLU の seed 1, 2（学内サーバ、自動で commit）"
git pull --rebase origin main && git push origin main
echo "all done $(date '+%F %T')"
