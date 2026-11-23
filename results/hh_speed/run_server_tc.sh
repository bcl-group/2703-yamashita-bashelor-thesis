#!/bin/bash
# 学内サーバ（V100）で、行列積の 3 方式（fp32 / fp16 / fp16x3 = Tensor Core）の速度と精度（Γ）を測る。
# RNN 1 個の学習（pid を引数で渡す）が終わるのを待ってから回し、最後に commit して push する。
# 使い方（リポジトリのルートで）: nohup bash results/hh_speed/run_server_tc.sh <待つ pid> > results/hh_speed/run_server_tc.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
if [ $# -ge 1 ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
export CUDA_VISIBLE_DEVICES=0
# RNN 1 個は、サーバにある同じ形の重み（sp_r_scr_e01_s1）で測る
M="hh_euler mm_rel mm_w256 k_3_w3 r8_mlp_hdf_s2 m3_v2f05d_s1 rnn3:p_h64_cl05 rnn3:s_h128_ft05 rnn3:s_gru64 rnn1:sp_r_scr_e01_s1 ens:mm_rel+mm_rel_s1"
PY="uv run python -u -X utf8 experiments/hh_surrogate/hh_speed.py"
# 1. 速度: 10 本の形 × 3 方式 × 2 つの回し方 × 5 通りの B
$PY bench --device cuda --models $M --mm fp32 fp16 fp16x3 --runner eager cudagraph \
    --B 1 32 1024 8192 65536 --steps 2000 --warmup 500 --repeats 5 --tag gpu_all
# 2. 精度: 26 本のテスト電流の Γ を 4 方式で（fp64 が基準）
$PY gamma --device cuda --models $M --mm fp64 fp32 fp16 fp16x3 --tag gpu
nvidia-smi > results/hh_speed/gpu_env.txt
git add experiments/hh_surrogate/hh_speed.py experiments/hh_surrogate/tc_matmul.py results/hh_speed
git commit -m "GPU（V100）での速度と精度: 行列積を fp32 / fp16 / fp16x3（Tensor Core）で比べた結果（学内サーバ、自動で commit）"
git pull --rebase origin hh-serial && git push origin hh-serial
