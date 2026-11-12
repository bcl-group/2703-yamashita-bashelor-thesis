#!/usr/bin/env bash
# 論文（Learning flow functions of spiking systems, L4DC 2024）と同じ方法で学習し、膜電位の応答を描く。
#
# 論文のコードは著者リポジトリの 2407-l4dc ブランチにある（master の flumen とは別物）。
# そのコードは一切書き換えず、コマンドライン引数で論文 §4 の設定を与えるだけにする。
# torch 2.14 で動かすため、論文コードの Python スクリプトは l4dc_compat.py 経由で起動する。
#
#   fs  : §4.1 速いスパイクの 1 細胞（図2）。N=800, T=500 ms, K=50000, LSTM 24
#   ffe : §4.2 2 細胞の順方向接続（図4）。N=200, T=1000 ms, K=50000, LSTM 32
#   共通: Δ=10 ms, 60/20/20 分割, 符号化器・復号器は隠れ層 3 層, 窓付き損失 (L=20, |J|<=5),
#         Adam lr=1e-3, 検証損失が 5 エポック下がらなければ lr を 1/10, 15 エポックで打ち切り,
#         モデルを N_MODELS 個（論文は 10 個）学習して検証損失が最小のものを使う
#
# 実行方法（卒論リポジトリのルートで）:
#     mkdir -p results/flow_learning
#     nohup bash experiments/flow_learning/run_paper.sh fs  > results/flow_learning/run_paper_fs.log  2>&1 &
#     nohup bash experiments/flow_learning/run_paper.sh ffe > results/flow_learning/run_paper_ffe.log 2>&1 &
# 学習するモデルの数を変えるとき: N_MODELS=1 bash experiments/flow_learning/run_paper.sh fs
# 途中で止まっても、終わった段階（環境・データ・学習済みモデルの数）は飛ばして続きから動く。
set -euo pipefail

TARGET="${1:-fs}"
N_MODELS="${N_MODELS:-10}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L4="$ROOT/external/flow-learning-l4dc"
PY="$L4/.venv/bin/python"
RUN="$PY $ROOT/experiments/flow_learning/l4dc_compat.py"   # 論文コードの起動に使う
RES="$ROOT/results/flow_learning"

case "$TARGET" in
    fs)  GEN=hhfs_flow.py;  TIME_HORIZON=5;  N_TRAJ=800; RNN=24; FIG_T=10 ;;
    ffe) GEN=hhffe_flow.py; TIME_HORIZON=10; N_TRAJ=200; RNN=32; FIG_T=14 ;;
    *) echo "使い方: $0 fs|ffe"; exit 1 ;;
esac
NAME="hh$TARGET"

mkdir -p "$RES"
echo "=== start $NAME: $(date '+%F %T') ==="

# ---- 1. 論文のコードを取得（書き換えない） ----
if [ ! -d "$L4" ]; then
    git clone -b 2407-l4dc https://github.com/mcpca/flow-learning "$L4"
fi
echo "l4dc commit: $(git -C "$L4" rev-parse --short HEAD)"

# ---- 2. 環境 ----
# torch 2.14 + CUDA 12.6 は V100（SM 7.0）で動く（2.15 + CUDA 13 は非対応）。
# pyDOE2 は Python 3.12 で消えた imp を使うので Python 3.11 にする。
# 完了の目印は「全部インストールできた」ときだけ作る。途中で失敗しても、再実行でインストールをやり直す。
ENV_OK="$L4/.venv/.deps_installed"
if [ ! -f "$ENV_OK" ]; then
    [ -x "$PY" ] || uv venv --python 3.11 "$L4/.venv"
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.0"
    uv pip install --python "$PY" numpy scipy pyDOE2 matplotlib
    touch "$ENV_OK"
fi

# GPU 上で LSTM が実際に動くかを、学習の前に確かめる（動かなければここで止まる）
"$PY" - <<'EOF'
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'cuDNN', torch.backends.cudnn.version())
print('GPU', torch.cuda.get_device_name(0), 'CC', torch.cuda.get_device_capability(0))
y, _ = torch.nn.LSTM(2, 8, batch_first=True).cuda()(torch.randn(4, 10, 2, device='cuda'))
print('GPU LSTM test: OK', tuple(y.shape))
EOF

cd "$L4"
export PYTHONPATH="$L4"
export PYTHONUNBUFFERED=1
mkdir -p outputs

# ---- 3. 教師データ（CPU） ----
# 保存先は outputs/<名前>.pth（論文コードの generate_data.py が決める）。分割は既定値 20/20（= 60/20/20）
if [ ! -f "outputs/$NAME.pth" ]; then
    echo "=== data generation: $(date '+%F %T') ==="
    $RUN "experiments/$GEN" "$NAME.pth" --control_delta 0.1 --time_horizon "$TIME_HORIZON" \
        --n_trajectories "$N_TRAJ" --n_samples 50000
fi

# ---- 4. 学習（GPU）を N_MODELS 回 ----
mkdir -p "outputs/$NAME"
while [ "$(ls "outputs/$NAME"/*.pth 2>/dev/null | wc -l)" -lt "$N_MODELS" ]; do
    echo "=== training $(( $(ls "outputs/$NAME"/*.pth 2>/dev/null | wc -l) + 1 ))/$N_MODELS: $(date '+%F %T') ==="
    $RUN experiments/load_data.py "outputs/$NAME.pth" \
        --control_rnn_size "$RNN" --encoder_depth 3 --decoder_depth 3 \
        --lr 1e-3 --sched_patience 5 --sched_factor 10 --es_patience 15 \
        --max_seq_len 20 --samples_per_state 5 \
        --experiment_id "$NAME"
done

# 検証損失が最小のモデル（論文コードの find_best_model.py）
BEST="$L4/$($RUN experiments/scripts/find_best_model.py "outputs/$NAME")"
echo "best model: $BEST"

# ---- 5. 損失の分布（図3） ----
# 論文コードの compute_stats.py が outputs/<名前>/<名前>_stats.csv（1 行 = 1 モデル）を書く。
# 作図は plot_loss_box.py（論文の l4dc_loss_plot.py は seaborn・LaTeX が要り、図を保存しないため）。
# 失敗しても予測の図（6.）は描けるように、ここでは止めない。
echo "=== loss stats: $(date '+%F %T') ==="
if $RUN experiments/scripts/compute_stats.py "outputs/$NAME"; then
    cp "outputs/$NAME/${NAME}_stats.csv" "$RES/${NAME}_stats.csv"
    MPLBACKEND=Agg "$PY" -X utf8 "$ROOT/experiments/flow_learning/plot_loss_box.py" \
        "$RES/${NAME}_stats.csv" --out "$RES/${NAME}_loss_box.png" --paper "$TARGET" \
        || echo "警告: 損失の箱ひげ図を描けなかった"
else
    echo "警告: compute_stats.py が失敗した（損失の分布は描かない）"
fi

# ---- 6. 膜電位の応答を描く（図2・図4） ----
echo "=== predict: $(date '+%F %T') ==="
cd "$ROOT"
for CUR in paper current_py; do
    MPLBACKEND=Agg "$PY" -X utf8 experiments/flow_learning/predict_current_py.py "$BEST" \
        --currents "$CUR" --out "$RES/${NAME}_${CUR}.png"
done

# 論文の作図スクリプトそのもの（LaTeX があるときだけ。図2・図4 と同じ見た目の PDF）
if command -v latex > /dev/null; then
    mkdir -p "$RES/${NAME}_l4dc_plot" && cd "$RES/${NAME}_l4dc_plot"
    MPLBACKEND=Agg $RUN "$L4/experiments/scripts/l4dc_plot_predictions.py" "$BEST" \
        --plot_input --n_plots 2 --time_horizon "$FIG_T"
fi

echo "=== finished $NAME: $(date '+%F %T') ==="
