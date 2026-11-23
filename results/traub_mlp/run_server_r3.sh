#!/bin/bash
# 学内サーバ（V100、GPU 0）で Traub の学習 3 回目。2 回目はノイズありの r2_256x3_d8_nz が最良（検証 500 ms で Γ 0.79、2 s で 0.70）。
# ノイズなしの 2 本は、DAgger の状態を GPU に溜めすぎてメモリ不足で止まった（プールを CPU に移して直した）
# 3 回目はノイズありを基本に:
#   - ノイズを 2 倍（V 1.0 mV、ゲート 0.01）
#   - 幅 512
#   - DAgger を 12 回
# 最後に検証用の 30 本を 2 s 回して採点（テスト用の軌道は使わない）
# 使い方（リポジトリのルートで）: nohup bash results/traub_mlp/run_server_r3.sh > results/traub_mlp/run_server_r3.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
PY="uv run python -u -X utf8"
T="$PY experiments/traub_surrogate/traub_mlp.py train --data teacher_v2 --device cuda --L 3 --iters 20000 --dagger-iters 10000 --dagger-n 140"
$T --tag r3_256x3_d8_nz1 --W 256 --dagger 8 --noise-v 1.0 --noise-g 0.01 > results/traub_mlp/r3_256x3_d8_nz1.log 2>&1 &
$T --tag r3_512x3_d8_nz --W 512 --dagger 8 --noise-v 0.5 --noise-g 0.005 > results/traub_mlp/r3_512x3_d8_nz.log 2>&1 &
$T --tag r3_256x3_d12_nz --W 256 --dagger 12 --noise-v 0.5 --noise-g 0.005 > results/traub_mlp/r3_256x3_d12_nz.log 2>&1 &
wait
for t in r3_256x3_d8_nz1 r3_512x3_d8_nz r3_256x3_d12_nz; do
    $PY experiments/traub_surrogate/traub_mlp.py eval --data teacher_v2 --device cuda --tag $t --split val --val-ms 2000 2>&1 | grep -v Warn | tail -n 1
done
git add results/traub_mlp/*.json results/traub_mlp/*.pt
git commit -m "Traub の学習 3 回目: ノイズあり（2 倍・幅 512・DAgger 12 回）（学内サーバ、自動で commit）"
git pull --rebase origin hh-serial && git push origin hh-serial
