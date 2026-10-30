#!/usr/bin/env bash
# flow-learning を GPU サーバーで HH FS に学習させ、current.py の電流で予測を描く。
#
# 1. external/flow-learning が無ければクローンし、データ生成のバグ（state_noise）を直す
# 2. external/flow-learning/.venv に Python 3.11 の環境を作る（uv を使う）
# 3. HH FS の教師データを生成（CPU）→ 学習（GPU）→ predict_current_py.py で描画
#
# 実行方法（卒論リポジトリのルートで）:
#     nohup bash experiments/flow_learning/run_server.sh > results/flow_learning/run_server.log 2>&1 &
# 途中で止まっても、終わった段階（環境・データ）は飛ばして続きから動く。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FL="$ROOT/external/flow-learning"
PY="$FL/.venv/bin/python"
NAME="hhfs_small"   # データ・学習結果の名前

# データの大きさ（run_hhfs.ps1 と同じ）。time_horizon はモデル時間（1 = 100 ms）
TIME_HORIZON=100      # 10 s
N_TRAJECTORIES=200
N_SAMPLES=5000

mkdir -p "$ROOT/results/flow_learning"
echo "=== start: $(date '+%F %T') ==="

# ---- 1. クローン ----
if [ ! -d "$FL" ]; then
    git clone https://github.com/mcpca/flow-learning "$FL"
    git -C "$FL" checkout 6f9aba3
    # 公式コードは存在しない data.state_noise を触って落ちるので、その行を消す
    sed -i '/data.state_noise\[k\] = /d' "$FL/experiments/semble_generate.py"
fi

# ---- 2. 環境 ----
if [ ! -x "$PY" ]; then
    uv venv --python 3.11 "$FL/.venv"
    # PyPI の semble は別物なので、著者の GitHub から入れる
    uv pip install --python "$PY" -e "$FL" torch scipy wandb pyyaml matplotlib \
        "semble @ git+https://github.com/mcpca/semble@366f3ef079c9b31116d6f46bc40ff3b8c927d317"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd "$FL"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ---- 3. 教師データ ----
if [ ! -f "data/$NAME.pkl" ]; then
    echo "=== data generation: $(date '+%F %T') ==="
    "$PY" experiments/semble_generate.py --time_horizon "$TIME_HORIZON" \
        --n_trajectories "$N_TRAJECTORIES" --n_samples "$N_SAMPLES" \
        data_generation/hhfs.yaml "$NAME"
fi

# ---- 4. 学習 ----
echo "=== training: $(date '+%F %T') ==="
"$PY" experiments/train_wandb.py "data/$NAME.pkl" "$NAME"

# 一番新しい学習結果のフォルダ
MODEL_DIR="$(ls -td "outputs/$NAME"/*/ | head -1)"
echo "model: $MODEL_DIR"

# ---- 5. current.py の電流で予測 ----
echo "=== predict: $(date '+%F %T') ==="
cd "$ROOT"
MPLBACKEND=Agg "$PY" -X utf8 experiments/flow_learning/predict_current_py.py "$FL/$MODEL_DIR" \
    --out "results/flow_learning/predict_current_py.png"

echo "=== finished: $(date '+%F %T') ==="
