#!/bin/bash
# 3 段階をまとめて回す: ブロックの学習 → つないだ評価 → 閉ループ微調整
# 使い方（リポジトリのルートで）: nohup bash results/hh_serial/run_all.sh > results/hh_serial/run_all.log 2>&1 &
set -e
D="$(dirname "$0")"
bash "$D/run_jobs.sh" jobs_1_blocks.txt 6
bash "$D/run_jobs.sh" jobs_2_eval.txt 5
bash "$D/run_jobs.sh" jobs_3_chain.txt 5
echo all done
