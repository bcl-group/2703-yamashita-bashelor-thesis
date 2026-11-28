#!/bin/bash
# 学内サーバ（V100、GPU 0）で Traub の学習 2 回目。1 回目（t_mlp256x3_dag: 検証 Γ 0.63、最後まで上がり続けた）を受けて
#   - DAgger を 3 回 → 8 回、1 回に回す学習用の軌道を 64 → 140 本（全部）
#   - 状態にノイズを足して正解を作り直す（--noise-v 0.5 mV, --noise-g 0.005）
#   - 幅 512
# 3 本を同時に回し、最後に検証用の 30 本を 2 s 回して採点（テスト用の軌道は使わない）
# 使い方（リポジトリのルートで）: nohup bash results/traub_mlp/run_server_r2.sh > results/traub_mlp/run_server_r2.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
PY="uv run python -u -X utf8"
T="$PY experiments/traub_surrogate/traub_mlp.py train --data teacher_v2 --device cuda --L 3 --iters 20000 --dagger 8 --dagger-iters 10000 --dagger-n 140"
$T --tag r2_256x3_d8 --W 256 > results/traub_mlp/r2_256x3_d8.log 2>&1 &
$T --tag r2_256x3_d8_nz --W 256 --noise-v 0.5 --noise-g 0.005 > results/traub_mlp/r2_256x3_d8_nz.log 2>&1 &
$T --tag r2_512x3_d8 --W 512 > results/traub_mlp/r2_512x3_d8.log 2>&1 &
wait
for t in t_mlp256x3_dag r2_256x3_d8 r2_256x3_d8_nz r2_512x3_d8; do
    $PY experiments/traub_surrogate/traub_mlp.py eval --data teacher_v2 --device cuda --tag $t --split val --val-ms 2000 2>&1 | grep -v Warn | tail -n 1
done
git add results/traub_mlp/*.json results/traub_mlp/*.pt
git commit -m "Traub の学習 2 回目: DAgger 8 回・ノイズ・幅 512（学内サーバ、自動で commit）"
git pull --rebase origin main && git push origin main
