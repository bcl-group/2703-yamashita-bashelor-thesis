#!/bin/bash
# 最終テスト（54 本）を、selection.json で選んだ 4 分野と、A の参考（seed 3 本ある最良の設定 r8_mlp_hdf）について 1 回だけ回す。
# 3 本ずつ並列（各 2 スレッド）。使い方（リポジトリのルートで）: bash results/final/run_final.sh > results/final/run_final.out 2>&1
cd "$(dirname "$0")/../.."
PY="uv run python -X utf8 results/eval_final.py --threads 2"
$PY r8_mlp_hd_w256 m3_hdf_s0 rnn3:p_h64_cl1 &
$PY sp_r_init sp_r_init_s1 sp_r_init_s2 &
$PY r8_mlp_hdf_s0 r8_mlp_hdf_s1 r8_mlp_hdf_s2 &
wait
echo ALL DONE
