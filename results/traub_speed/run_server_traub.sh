#!/bin/bash
# 学内サーバ（V100、GPU 0）で Traub の土台を回す: 1. 速度 2. 教師データ（200 軌道 × 2 s）3. 学習 3 本 4. commit・push
# 使い方（リポジトリのルートで）: nohup bash results/traub_speed/run_server_traub.sh > results/traub_speed/run_server_traub.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
PY="uv run python -u -X utf8"
# 1. 速度（乱数の重み）。ODE と代理モデルの形、行列積 3 方式、2 つの回し方
$PY experiments/traub_surrogate/traub_speed.py --device cuda --mm fp32 fp16 fp16x3 --runner eager cudagraph \
    --B 1 32 1024 8192 32768 --steps 2000 --warmup 500 --repeats 5 --tag gpu
# 2. 教師データ
$PY experiments/traub_surrogate/make_dataset_v2.py --n 200 --T 2000 --device cuda
# 3. 学習（2 本ずつ並列）
T="$PY experiments/traub_surrogate/traub_mlp.py train --data teacher_v2 --device cuda"
$T --tag t_mlp128 --iters 30000 > results/traub_speed/t_mlp128.log 2>&1 &
$T --tag t_mlp128_dag --iters 20000 --dagger 3 --dagger-iters 10000 > results/traub_speed/t_mlp128_dag.log 2>&1 &
wait
$T --tag t_mlp256x3_dag --W 256 --L 3 --iters 20000 --dagger 3 --dagger-iters 10000 > results/traub_speed/t_mlp256x3_dag.log 2>&1
# 4. commit・push（教師データは大きいので入れない）
git add results/traub_speed results/traub_mlp/*.json results/traub_mlp/*.pt
git commit -m "Traub の土台: GPU での速度（ODE と代理モデルの形）と、区画ごとの Markov MLP の学習 3 本（学内サーバ、自動で commit）"
git pull --rebase origin hh-serial && git push origin hh-serial
