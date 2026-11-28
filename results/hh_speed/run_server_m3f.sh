#!/bin/bash
# 学内サーバ（V100）で、C（m3_hdf_s0）を Tensor Core に載せた版の速度・精度・カーネルを測り、結果を commit・push する。
#   m3_hdf_s0       元の C（--mm fp16 のときは tc_matmul で nn.Linear だけ fp16 にする）
#   m3f:m3_hdf_s0   行列積だけで回す版（m3_fused）。fp32 = FusedM3、fp16 = FusedM3Half（Tensor Core 版）
#   m3one:m3_hdf_s0 1 本の 6 層ネットワークにした版（m3_fused.FusedM3One。各層が行列積 1 回、重みはブロック対角）
# 使い方（リポジトリのルートで）:
#   git pull origin hh-serial
#   nohup bash results/hh_speed/run_server_m3f.sh > results/hh_speed/logs/run_server_m3f.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p results/hh_speed/logs
export CUDA_VISIBLE_DEVICES=0
M="hh_euler m3_hdf_s0 m3f:m3_hdf_s0 m3one:m3_hdf_s0"
PY="uv run python -u -X utf8 experiments/hh_surrogate"
L=results/hh_speed/logs
echo "start $(date '+%F %T')"
nvidia-smi > results/hh_speed/gpu_env_m3f.txt
# 0. 元の閉ループと一致するか（fp64、CPU）
$PY/hh_speed.py check --models m3_hdf_s0 m3f:m3_hdf_s0 m3f:m3_hdf_relu_s0 m3one:m3_hdf_s0 m3one:m3_hdf_relu_s0 2>&1 | tee $L/m3f_check.log
# 1. どのカーネルが動くか（Tensor Core のカーネル名が出るか）
$PY/m3_prof.py --B 65536 --mm fp32 fp16 2>&1 | tee $L/m3f_prof.log
$PY/m3_prof.py --B 1024 --mm fp32 fp16 2>&1 | tee -a $L/m3f_prof.log
# 2. 速度: fp32 / fp16 × 2 つの回し方 × 5 通りの B（gpu_all と同じ条件）
$PY/hh_speed.py bench --device cuda --models $M --mm fp32 fp16 --runner eager cudagraph \
    --B 1 32 1024 8192 65536 --steps 2000 --warmup 500 --repeats 5 --tag gpu_m3f 2>&1 | tee $L/gpu_m3f.log
# 3. torch.compile（要素ごとの演算の融合）+ CUDA Graph。V100 で Inductor が通らなければ失敗するだけ
$PY/hh_speed.py bench --device cuda --models m3f:m3_hdf_s0 m3one:m3_hdf_s0 --mm fp32 fp16 --runner cudagraph --compile \
    --B 1024 8192 65536 --steps 2000 --warmup 500 --repeats 5 --tag gpu_m3f_compile 2>&1 | tee $L/gpu_m3f_compile.log \
    || echo "compile は失敗（ログを見る）"
# 4. 精度: 26 本の開発用テスト電流の Γ（fp64 が基準）
$PY/hh_speed.py gamma --device cuda --models m3_hdf_s0 m3f:m3_hdf_s0 m3one:m3_hdf_s0 --mm fp64 fp32 fp16 --tag gpu_m3f 2>&1 | tee -a $L/gpu_m3f.log
git add results/hh_speed/gpu_env_m3f.txt results/hh_speed/gpu_m3f*.json results/hh_speed/gpu_m3f*.png \
    results/hh_speed/gamma_gpu_m3f.json results/hh_speed/gamma_gpu_m3f $L/m3f_*.log $L/gpu_m3f*.log
git commit -m "C を Tensor Core に載せた版（m3_fused）の V100 での速度・精度・カーネル（学内サーバ、自動で commit）"
git pull --rebase origin hh-serial && git push origin hh-serial
echo "all done $(date '+%F %T')"
