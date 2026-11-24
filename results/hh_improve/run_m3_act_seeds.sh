#!/bin/bash
# C（MLP 3 個、m3_hdf）の活性化関数の比較: ReLU と SiLU の seed 1, 2 を、ローカルの CPU で 1 本ずつ順に学習する
# （メモリの空きが少ないので並列にしない）。条件は m3_hdf_s0（jobs_gpu_round8.txt の 29 行目）と同じで、--act と --seed だけ変える。
# 使い方（リポジトリのルートで）: bash results/hh_improve/run_m3_act_seeds.sh
cd "$(dirname "$0")/../.."
mkdir -p results/hh_improve/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
COMMON="--sfeat --hard 256 --hard-ms 3000 --hard-stride 10 --dagger 3 --dagger-iters 20000 --dagger-n 256 --dagger-ms 1000 --dagger-stride 10 --H 128 --layers 2 --bank 2048 --ep-ms 100 --pre-batch 1024 --pre-iters 50000 --pre-rel 0.02 --device cpu --threads 8"
for job in "relu 1 m3_hdf_relu_s1" "silu 1 m3_hdf_s1" "relu 2 m3_hdf_relu_s2" "silu 2 m3_hdf_s2"; do
    set -- $job
    act=$1; seed=$2; tag=$3
    if [ -f "results/hh_improve/$tag.json" ]; then
        echo "skip $tag（学習済み）"
        continue
    fi
    echo "start $tag: $(date '+%F %T')"
    .venv/Scripts/python.exe -u -X utf8 experiments/hh_surrogate/hh_improve.py mlp3 --tag "$tag" --act "$act" --seed "$seed" $COMMON \
        > "results/hh_improve/logs/$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAILED $tag (exit $rc)"
    else
        echo "done $tag: $(grep 'val best' results/hh_improve/logs/$tag.log | tail -1 | sed -E 's/不一致.*//')"
    fi
done
echo "all done $(date '+%F %T')"
