# HH 代理モデルの一覧を Excel にまとめる（results/HHモデル一覧.xlsx）
# 使い方（リポジトリのルートで）: uv run python -X utf8 results/make_model_list.py
import json
import os

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HHモデル一覧.xlsx')
GAMMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gamma')   # rescore_gamma.py の結果
N_TEST, N_VAL = 26, 17

# (分類, タグ, 構造, 学習, 検証17本, テスト26本, 平均一致率, テストで外れた電流, 備考, 結果ファイル)
# None は未評価
ROWS = [
    # ---- RNN 3 個（分割型）----
    ('RNN 3 個', 'hh_modular_ou', 'RNN1(V→α,β) / RNN2(α,β,g→Δg) / RNN3(g,V,I→ΔV)、H=50、tanh RNN',
     '教師強制、150 エポック（numpy 版）', 11, 19, 0.927,
     'const 7, ramp 0-20, ramp 0-30, random pulse 5ms, OU 5±3, OU 10±3, level+OU',
     '基準。卒論の元のモデル 2', 'results/hh_modular_ou/'),
    ('RNN 3 個', 'a3_h64x2', 'RNN3 を H64×2 層に（RNN1・RNN2 は基準のまま）', '教師強制、3000 回',
     None, 15, 0.824, 'const 7, const 10, ramp 0-20, sin 10Hz など 11 本',
     '基準の RNN3 を深くしたやつ（第1ラウンドで最良）', 'results/hh_improve/a3_h64x2.pt'),
    ('RNN 3 個', 'a3_h128', 'RNN3 を H128 に', '教師強制、3000 回', None, 13, 0.737,
     '13 本', '基準の RNN3 を広くしたやつ', 'results/hh_improve/a3_h128.pt'),
    ('RNN 3 個', 'a3_h64_feat', 'RNN3 の入力に m³h, n⁴, m³hV, n⁴V を追加', '教師強制、3000 回', None, 11, 0.733,
     '15 本', 'RNN3 に積の特徴量を足したやつ', 'results/hh_improve/a3_h64_feat.pt'),
    ('RNN 3 個', 'a3_h64', 'RNN3 を H64 に', '教師強制、3000 回', None, 9, 0.687,
     '17 本', 'PyTorch で学習し直した比較用', 'results/hh_improve/a3_h64.pt'),
    ('RNN 3 個', 'a3_h64_noise001', 'RNN3 H64', '教師強制 + 入力の雑音 0.01（正解は式から再計算）', None, 9, 0.689,
     '17 本', '入力に雑音を足したやつ', 'results/hh_improve/a3_h64_noise001.pt'),
    ('RNN 3 個', 'a3_h64_noise01', 'RNN3 H64', '教師強制 + 入力の雑音 0.1', None, 9, 0.675,
     '17 本', '入力に雑音を足したやつ（強め）', 'results/hh_improve/a3_h64_noise01.pt'),
    ('RNN 3 個', 'a3_h64_wsub10', 'RNN3 H64', '教師強制、閾値下（-70〜-40 mV）の損失×10', None, 9, 0.628,
     '17 本', '閾値下を重視したやつ', 'results/hh_improve/a3_h64_wsub10.pt'),
    ('RNN 3 個', 'a3_h64_nomem', 'RNN3 H64、再帰の重みを 0 から＋罰則', '教師強制', None, 8, 0.673,
     '18 本', 'RNN3 の記憶を抑えたやつ', 'results/hh_improve/a3_h64_nomem.pt'),
    ('RNN 3 個', 'c123_lr1e-6', '基準の 3 ブロック', '閉ループ微調整 K=1 ms、3 ブロックとも、lr 1e-6', None, 21, 0.939,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, OU 5±3',
     '基準を閉ループで微調整したやつ。※最良の時点を 26 本で選んだので参考値', 'results/hh_improve/c123_lr1e-6_b*.pt'),
    ('RNN 3 個', 'k_3_w3', '基準の 3 ブロック', '閉ループ微調整 K=1→5→20 ms、RNN3 のみ、閾値付近×3、lr 1e-6', 13, 22, 0.924,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms',
     '基準を閉ループで微調整したやつ（RNN 3 個型の最良）', 'results/hh_improve/k_3_w3_b3.pt'),
    ('RNN 3 個', 'k_123_w3', '基準の 3 ブロック', '閉ループ微調整 K=1→5→20 ms、3 ブロックとも、閾値付近×3', 12, 21, 0.939,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, OU 5±3', '3 ブロックとも動かすと不安定',
     'results/hh_improve/k_123_w3_b*.pt'),
    ('RNN 3 個', 'k_123_w1', '基準の 3 ブロック', '同上、閾値付近の重みなし', 12, 21, 0.939,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, OU 5±3', '', 'results/hh_improve/k_123_w1_b*.pt'),
    ('RNN 3 個', 'k_3_w3_K2000', 'k_3_w3 の RNN3', 'k_3_w3 から K=20 ms で 300 回、lr 5e-7', 13, 22, 0.924,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms', 'k_3_w3 をさらに長い窓で微調整したやつ（変化なし）',
     'results/hh_improve/k_3_w3_K2000_b3.pt'),
    ('RNN 3 個（直列）', 'e_s', 'RNN1(V) → RNN2 → RNN3(g(t+dt), V, I → ΔV) を同じ時刻で直列、各 H64',
     '教師強制、各 3000 回（微調整前）', 2, 6, 0.576, '20 本',
     'ホワイトボードのやつ（別チャットの図）', 'results/hh_serial/e_s.json'),
    ('RNN 3 個（直列）', 'e_s_i', '同上、RNN1 に I も入れる', '同上', 3, 6, 0.610, '20 本',
     'ホワイトボードのやつ。RNN1 に I を入れても変わらない', 'results/hh_serial/e_s_i.json'),
    ('RNN 3 個（直列）', 'e_s_g', '同上、RNN3 がコンダクタンス G_Na, G_K も出す（補助出力）', '同上', 3, 11, 0.670,
     'step 10, const 7・10・20, ramp 0-20・0-30, sin 10Hz など 15 本',
     'ホワイトボードのやつ。G も出させると +5 本（直列型で最良）', 'results/hh_serial/e_s_g.json'),
    ('RNN 3 個（直列）', 'e_s_ig', '同上、RNN1 に I + RNN3 が G も出す', '同上', 3, 10, 0.655, '16 本',
     'ホワイトボードのやつ', 'results/hh_serial/e_s_ig.json'),
    ('RNN 3 個（直列）', 'e_p', '比較用: 同じコードで RNN3 に g(t) を入れる（従来の並列）', '同上', 3, 7, 0.572, '19 本',
     'ホワイトボードのやつの比較用。直列と並列の差はほぼない', 'results/hh_serial/e_p.json'),
    ('RNN 3 個（直列）', 'c_s', 'e_s と同じ', 'e_s から閉ループ微調整 K=1→5→20 ms、3 ブロックとも、lr 1e-6（学内サーバ）',
     4, 7, 0.639, '19 本', 'ホワイトボードのやつを閉ループで微調整したやつ（6 → 7 本）',
     'results/hh_serial/c_s_b*.pt'),
    ('RNN 3 個（直列）', 'c_s_i', 'e_s_i と同じ', 'e_s_i から閉ループ微調整（同上、学内サーバ）',
     4, 6, 0.626, '20 本', 'ホワイトボードのやつを閉ループで微調整したやつ（上がらない）',
     'results/hh_serial/c_s_i_b*.pt'),
    ('RNN 3 個（直列）', 'c_s_g', 'e_s_g と同じ', 'e_s_g から閉ループ微調整（同上、学内サーバ）',
     3, 10, 0.720, 'step 10, const 7・10・20, ramp 0-20・0-30, sin 10Hz など 16 本',
     'ホワイトボードのやつ（G も出す版）を微調整したやつ（11 → 10 本。直列型で最良は微調整前の e_s_g）',
     'results/hh_serial/c_s_g_b*.pt'),
    ('RNN 3 個（直列）', 'c_s_ig', 'e_s_ig と同じ', 'e_s_ig から閉ループ微調整（同上、学内サーバ）',
     3, 10, 0.687, '16 本', 'ホワイトボードのやつを微調整したやつ（10 → 10 本）', 'results/hh_serial/c_s_ig_b*.pt'),
    ('RNN 3 個（直列）', 'c_p', 'e_p と同じ', 'e_p から閉ループ微調整（同上、学内サーバ）', 4, 6, 0.591, '20 本',
     '比較用を閉ループで微調整したやつ（上がらない）', 'results/hh_serial/c_p_b*.pt'),
    # ---- RNN 1 個（再帰あり）----
    ('RNN 1 個', 's_h64', '[V, I] → ΔV、tanh RNN H64（ゲートは隠れ状態）', '教師強制', None, 0, 0.215,
     '26 本すべて', '電流が切れても発火し続ける', 'results/hh_improve/s_h64.pt'),
    ('RNN 1 個', 's_h64_aux', '同上 + 隠れ状態から m,h,n を当てる補助出力', '教師強制', None, 0, 0.058,
     '26 本すべて', '補助出力を足したやつ', 'results/hh_improve/s_h64_aux.pt'),
    ('RNN 1 個', 's_h64_aux_leak', '同上 + 漏れ積分型の隠れ状態', '教師強制', None, 3, 0.283,
     '23 本', '漏れ積分にしたやつ', 'results/hh_improve/s_h64_aux_leak.pt'),
    ('RNN 1 個', 's_h64_ro200', 's_h64 と同じ', 's_h64 から rollout 2 ms で微調整', None, 0, 0.130,
     '26 本すべて', 's_h64 を rollout で改善しようとしたやつ', 'results/hh_improve/s_h64_ro200.pt'),
    ('RNN 1 個', 's_mix_h64_aux_r', 's_h64_aux と同じ', '教師強制、電流の切り替えが多いデータを混ぜる', None, 0, 0.050,
     '26 本すべて', '止まれるようになったが OU で発散', 'results/hh_improve/s_mix_h64_aux_r.pt'),
    ('RNN 1 個', 's_mix_h128_aux_r', '同上、H128', '同上', None, 0, 0.092,
     '26 本すべて', '', 'results/hh_improve/s_mix_h128_aux_r.pt'),
    ('RNN 1 個', 'cl_v_rnn / cl_v_gru', '[V, I] → ΔV、tanh RNN / GRU', '閉ループが主（静止状態から、α 1→0）', 1, None, None,
     '未評価（途中で中断）', '閉ループで学習し直したやつ', 'results/hh_improve/cl_v_*.pt'),
    ('RNN 1 個', 'cl_s_rnn / cl_s_rnn_h128 / cl_s_gru / cl_s_rnn_noonpol',
     '[V, m, h, n, I] → [ΔV, Δm, Δh, Δn]、予測した m,h,n も戻す', '閉ループが主 + 式で計算した正解との損失', 1, None, None,
     '未評価（途中で中断）', 'RNN 1 個で m,h,n まで予測して戻すやつ（ユーザー案）', 'results/hh_improve/cl_s_*.pt'),
    # ---- Markov MLP ----
    ('Markov MLP', 'cl_m_mlp', '[V, m, h, n, I] → [ΔV, Δm, Δh, Δn]、MLP 128×2（再帰なし）',
     '摂動した状態で 1 ステップ事前学習 2 万回 → 閉ループ微調整', 5, 10, 0.737, '16 本',
     'RNN 1 個型から再帰を外したやつ（Codex 案）', 'results/hh_improve/cl_m_mlp.pt'),
    ('Markov MLP', 'mm_base', '同上', '事前学習 5 万回、教師データを 2048 本×100 ms に', 12, 20, 0.910,
     'ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, OU 5±3, level+OU',
     'cl_m_mlp の事前学習を長く・データを多くしたやつ', 'results/hh_improve/mm_base.pt'),
    ('Markov MLP', 'mm_rel', '同上', 'mm_base の ΔV の損失を相対誤差（増分の大きさで割る）に', 15, 25, 0.962,
     'ramp 0-30', 'mm_base の損失を改善したやつ（全体で最良）', 'results/hh_improve/mm_rel.pt'),
    ('Markov MLP', 'mm_sv05', '同上', 'mm_rel の摂動を 1/4（0.5 mV / 0.005）', 12, 22, 0.954,
     'const 7, ramp 0-20, ramp 0-30, munechika train', 'mm_rel の摂動を小さくしたやつ', 'results/hh_improve/mm_sv05.pt'),
    ('Markov MLP', 'mm_sv5', '同上', 'mm_rel の摂動を ×2.5（5 mV / 0.05）', 12, 20, 0.919,
     'const 7, ramp 0-20, ramp 0-30, sin 10Hz, level+OU, munechika train', 'mm_rel の摂動を大きくしたやつ',
     'results/hh_improve/mm_sv5.pt'),
    ('Markov MLP', 'mm_w256', 'MLP 256×3', 'mm_rel と同じ', 11, 23, 0.950,
     'const 7, ramp 0-20, munechika train', 'mm_rel を広く・深くしたやつ', 'results/hh_improve/mm_w256.pt'),
    ('Markov MLP', 'mm_w256_long', 'MLP 256×3', 'mm_w256 の事前学習を 15 万回に', 16, 23, 0.941,
     'ramp 0-20, ramp 0-30, OU 5±3', 'mm_w256 を長く学習したやつ（検証では最良）', 'results/hh_improve/mm_w256_long.pt'),
    ('Markov MLP', 'mm_rel_s1', 'MLP 128×2、SiLU', 'mm_rel と同じ、seed 1', 11, 22, 0.921,
     'ramp 0-20, ramp 0-30, OU 5±3, munechika train', 'mm_rel の seed を変えたやつ（seed で 3 本動く）',
     'results/hh_improve/mm_rel_s1.pt'),
    ('Markov MLP', 'mm_rel_relu', 'MLP 128×2、ReLU', 'mm_rel と同じ（活性化関数だけ ReLU）', 11, 20, 0.882,
     'const 7, const 10, ramp 0-20, ramp 0-30, OU 5±3, munechika train', 'mm_rel を ReLU にしたやつ（exp なし）',
     'results/hh_improve/mm_rel_relu.pt'),
    ('Markov MLP', 'mm_rel_relu_s1', 'MLP 128×2、ReLU', '同上、seed 1', 14, 19, 0.883,
     'const 20, ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, OU 5±3, level+OU', 'mm_rel_relu の seed を変えたやつ',
     'results/hh_improve/mm_rel_relu_s1.pt'),
    ('Markov MLP', 'mm_w256_relu', 'MLP 256×3、ReLU', 'mm_w256 と同じ（活性化関数だけ ReLU）', 11, 20, 0.910,
     'const 7, ramp 0-20, ramp 0-30, sin 10Hz, random pulse 5ms, munechika train', 'mm_w256 を ReLU にしたやつ',
     'results/hh_improve/mm_w256_relu.pt'),
    # ---- ハイブリッド ----
    ('ハイブリッド', 'cl_p_mlp', 'MLP 64×2 は V → g∞, q（各ゲート）だけ。Δg = q(g∞−g)、ΔV は HH の電流の式',
     '摂動した状態で 1 ステップ事前学習 2 万回', 17, 23, 0.957, 'ramp 0-30, random pulse 5ms, OU 5±3',
     'HH の電流の式を組み込んだやつ（Codex 案）。純粋な代理モデルではない', 'results/hh_improve/cl_p_mlp.pt'),
    # ---- flow function ----
    ('flow function', 'hhfs（学内サーバ 10 本）', '論文（Aguiar ら 2024）の encoder + control-RNN + decoder',
     '論文のコードのまま（論文版の HH、0〜1 µA、solve_ivp）', None, None, None,
     '別の 7 電流で RMSE 4〜16 mV', 'flow function のやつ。対象の HH と評価が違うので直接は比べられない',
     'results/flow_learning/'),
]

# タグ → 学習・評価に使ったソースコード（スクリプトとサブコマンド）。引数は results/*/jobs_*.txt を参照
S = 'experiments/hh_surrogate/'
SOURCE = {
    'hh_modular_ou': S + 'hh_modular.py（train --block 1〜3 --target delta → eval --clip）',
    'a3_h64x2': S + 'hh_improve.py block --block 3',
    'a3_h128': S + 'hh_improve.py block --block 3',
    'a3_h64_feat': S + 'hh_improve.py block --block 3 --feat',
    'a3_h64': S + 'hh_improve.py block --block 3',
    'a3_h64_noise001': S + 'hh_improve.py block --block 3 --noise',
    'a3_h64_noise01': S + 'hh_improve.py block --block 3 --noise',
    'a3_h64_wsub10': S + 'hh_improve.py block --block 3 --wsub',
    'a3_h64_nomem': S + 'hh_improve.py block --block 3 --whh0 --whh-pen',
    'c123_lr1e-6': S + 'hh_improve.py chain --train 123',
    'k_3_w3': S + 'hh_improve.py chain --train 3 --K-sched',
    'k_123_w3': S + 'hh_improve.py chain --train 123 --K-sched',
    'k_123_w1': S + 'hh_improve.py chain --train 123 --K-sched',
    'k_3_w3_K2000': S + 'hh_improve.py chain --train 3 --b3 k_3_w3_b3.pt',
    'e_s': S + 'hh_serial.py（block → eval）',
    'e_s_i': S + 'hh_serial.py（block --i1 → eval）',
    'e_s_g': S + 'hh_serial.py（block --gout → eval）',
    'e_s_ig': S + 'hh_serial.py（block --i1 --gout → eval）',
    'e_p': S + 'hh_serial.py（block --conn parallel → eval）',
    'c_s': S + 'hh_serial.py chain',
    'c_s_i': S + 'hh_serial.py chain',
    'c_s_g': S + 'hh_serial.py chain',
    'c_s_ig': S + 'hh_serial.py chain',
    'c_p': S + 'hh_serial.py chain',
    's_h64': S + 'hh_improve.py single',
    's_h64_aux': S + 'hh_improve.py single --aux',
    's_h64_aux_leak': S + 'hh_improve.py single --aux --leak',
    's_h64_ro200': S + 'hh_improve.py single --rollout --init',
    's_mix_h64_aux_r': S + 'hh_improve.py single --aux（--data に 2 ファイル）',
    's_mix_h128_aux_r': S + 'hh_improve.py single --aux（--data に 2 ファイル）',
    'cl_v_rnn / cl_v_gru': S + 'hh_improve.py single_cl --cell rnn / gru',
    'cl_s_rnn / cl_s_rnn_h128 / cl_s_gru / cl_s_rnn_noonpol': S + 'hh_improve.py single_cl --state',
    'cl_m_mlp': S + 'hh_improve.py single_cl --state --cell mlp --pre-iters',
    'mm_base': S + 'hh_improve.py single_cl --state --cell mlp --pre-iters',
    'mm_rel': S + 'hh_improve.py single_cl --state --cell mlp --pre-iters --pre-rel',
    'mm_sv05': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --pre-sv --pre-sg',
    'mm_sv5': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --pre-sv --pre-sg',
    'mm_w256': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --H 256 --layers 3',
    'mm_w256_long': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --H 256 --layers 3',
    'cl_p_mlp': S + 'hh_improve.py single_cl --phys --cell mlp --pre-iters',
    'mm_rel_s1': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --seed 1',
    'mm_rel_relu': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --act relu',
    'mm_rel_relu_s1': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --act relu --seed 1',
    'mm_w256_relu': S + 'hh_improve.py single_cl --state --cell mlp --pre-rel --act relu --H 256 --layers 3',
    'hhfs（学内サーバ 10 本）': 'experiments/flow_learning/run_paper.sh fs（external/flow-learning-l4dc の論文コード）、'
                            '評価は predict_current_py.py',
}

HEAD = ['No', '分類', 'モデル（タグ）', '構造', '学習方法', '検証 17 本\n100% 一致', 'テスト 26 本\n100% 一致',
        'テスト\n一致した割合', 'テスト\n平均一致率', 'テスト\n平均 Γ', 'テスト順位\n(100% 一致)',
        'テスト順位\n(平均 Γ)', '追加: seed\n10 本ずつ Γ', '追加: f–I\nΓ', '追加: f–I\n発火率誤差 [Hz]',
        '追加: 長時間\n20 s Γ', '追加: 分布の\n外 Γ', 'テストで外れた電流', '備考', 'ソースコード', '結果ファイル']
WIDTH = [5, 14, 26, 46, 40, 11, 11, 11, 10, 10, 11, 11, 11, 10, 12, 11, 11, 40, 44, 48, 36]
EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext')   # eval_ext.py の結果
EXT_COLS = [('seeds', 'gamma'), ('fi', 'gamma'), ('fi', 'rate_rmse_hz'), ('long', 'gamma'), ('ood', 'gamma')]


def ext_of(tag):
    # results/ext/<タグ>.json（追加テスト）から EXT_COLS の値を読む。採点していないものは None
    p = os.path.join(EXT_DIR, tag + '.json')
    if not os.path.exists(p):
        return [None] * len(EXT_COLS)
    d = json.load(open(p, encoding='utf-8'))
    return [d[s]['summary'][k] if s in d else None for s, k in EXT_COLS]
FILL = {'RNN 3 個': 'DCE6F1', 'RNN 3 個（直列）': 'FDE9D9', 'RNN 1 個': 'F2DCDB', 'Markov MLP': 'EBF1DE',
        'ハイブリッド': 'EDEDED', 'flow function': 'FFF2CC'}

wb = Workbook()
ws = wb.active
ws.title = 'モデル一覧'
font = Font(name='Arial', size=10)
bold = Font(name='Arial', size=10, bold=True, color='FFFFFF')
thin = Side(style='thin', color='BFBFBF')
border = Border(left=thin, right=thin, top=thin, bottom=thin)

ws['A1'] = 'HH 代理モデルの一覧（26 本のテスト電流でのスパイク時刻の一致、±2 ms）'
ws['A1'].font = Font(name='Arial', size=12, bold=True)
ws['A2'] = ('100% 一致 = その電流のスパイクがすべて ±2 ms 以内で一致した本数。検証 17 本は最良の時点を選ぶのに使った別の電流。'
            '平均 Γ = coincidence factor（Kistler et al. 1997; Jolivet et al. 2008、Δ = 2 ms）の 26 本の平均。'
            '「—」は未評価。一致率は実験ログ、Γ は results/rescore_gamma.py で採点し直した値')
ws['A2'].font = Font(name='Arial', size=9, italic=True)
r0 = 4
for j, (h, w) in enumerate(zip(HEAD, WIDTH), 1):
    c = ws.cell(r0, j, h)
    c.font = bold
    c.fill = PatternFill('solid', fgColor='404040')
    c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    c.border = border
    ws.column_dimensions[get_column_letter(j)].width = w
ws.row_dimensions[r0].height = 32

def gamma_of(tag):
    """
    results/gamma/<タグ>.json（results/rescore_gamma.py の出力）から 26 本の平均 Γ を読む。
    「a / b」のようにまとめた行は、それぞれの値を「/」でつないだ文字列にする。採点していなければ None
    """
    vs = []
    for t in tag.split(' / '):
        p = os.path.join(GAMMA_DIR, t.strip() + '.json')
        if not os.path.exists(p):
            return None
        vs.append(json.load(open(p, encoding='utf-8'))['gamma'])
    return vs[0] if len(vs) == 1 else ' / '.join('{:.3f}'.format(v) for v in vs)


first, last = r0 + 1, r0 + len(ROWS)
for i, row in enumerate(ROWS):
    r = first + i
    cat, tag, arch, train, val, test, mean, fails, note, path = row
    gam = gamma_of(tag)
    ext = ext_of(tag)
    vals = [i + 1, cat, tag, arch, train, val, test, None, mean, gam, None, None, *ext, fails, note, SOURCE[tag],
            path]
    for j, v in enumerate(vals, 1):
        c = ws.cell(r, j, v if v is not None else None)
        c.font = font
        c.border = border
        c.fill = PatternFill('solid', fgColor=FILL[cat])
        c.alignment = Alignment(vertical='top', wrap_text=True,
                                horizontal='center' if j <= 12 + len(EXT_COLS) and j not in (2, 3, 4, 5) else 'left')
    for k, (v, (s, key)) in enumerate(zip(ext, EXT_COLS)):
        c = ws.cell(r, 13 + k)
        if v is None:
            c.value = '—'
        else:
            c.number_format = '0.0' if key == 'rate_rmse_hz' else '0.000'
    if val is None:
        ws.cell(r, 6, '—')
    if test is None:
        ws.cell(r, 7, '—')
        ws.cell(r, 8, '—')
        ws.cell(r, 11, '—')
    else:
        ws.cell(r, 8, f'=G{r}/{N_TEST}').number_format = '0%'
        ws.cell(r, 11, f'=IF(ISNUMBER(G{r}),RANK(G{r},$G${first}:$G${last},0),"—")')
    if mean is None:
        ws.cell(r, 9, '—')
    else:
        ws.cell(r, 9).number_format = '0.000'
    if gam is None:
        ws.cell(r, 10, '—')
        ws.cell(r, 12, '—')
    elif isinstance(gam, float):
        ws.cell(r, 10).number_format = '0.000'
        ws.cell(r, 12, f'=IF(ISNUMBER(J{r}),RANK(J{r},$J${first}:$J${last},0),"—")')
    else:
        ws.cell(r, 12, '—')
    if test is not None and test >= 22:
        ws.cell(r, 7).font = Font(name='Arial', size=10, bold=True)

ws.cell(first + 9, 7).comment = Comment('最良の時点を 26 本の成績で選んだので参考値', 'Claude')
ws.freeze_panes = ws.cell(r0 + 1, 4)
ws.auto_filter.ref = f'A{r0}:{get_column_letter(len(HEAD))}{last}'

# 凡例
lr = last + 2
ws.cell(lr, 2, '色の凡例').font = Font(name='Arial', size=10, bold=True)
for k, (cat, col) in enumerate(FILL.items()):
    c = ws.cell(lr + 1 + k, 2, cat)
    c.font = font
    c.fill = PatternFill('solid', fgColor=col)
    c.border = border
ws.cell(lr + 1, 4, 'テスト順位は 100% 一致の本数、または平均 Γ の大きい順（同じ値は同順位）。一致した割合 = 100% 一致の本数 / 26').font = font
ws.cell(lr + 2, 4, '追加テスト（experiments/hh_surrogate/hh_bench_ext.py、上位 8 モデルのみ）: '
                   'seed 10 本ずつ = OU 3 種・ランダムステップ・ランダムパルスを seed 10 本ずつ（50 本、各 500 ms）の平均 Γ。'
                   'f–I = 定常電流 0〜30 を 0.5 刻み（61 本、各 1 s）の平均 Γ と、後半 500 ms の発火率の誤差（RMSE）。'
                   '長時間 = OU と段階 + OU を各 20 s（6 本）の平均 Γ。'
                   '分布の外 = 強い抑制・強い入力・チャープ・とても遅いランプなど 8 本の平均 Γ').font = font

# ---------------- 電流ごとの結果（シートを分ける） ----------------
from openpyxl.formatting.rule import ColorScaleRule   # noqa: E402

ORDER = [r[1] for r in ROWS]   # 一覧と同じ並び


def tags_in(dir_):
    # dir_ にある <タグ>.json を、一覧の並び（一覧にないものは末尾）で返す
    have = [f[:-5] for f in os.listdir(dir_) if f.endswith('.json')]
    return [t for t in ORDER if t in have] + sorted(t for t in have if t not in ORDER)


def header(ws_, cols, widths):
    for j, (h, w) in enumerate(zip(cols, widths), 1):
        c = ws_.cell(1, j, h)
        c.font = bold
        c.fill = PatternFill('solid', fgColor='404040')
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws_.column_dimensions[get_column_letter(j)].width = w
    ws_.row_dimensions[1].height = 30
    ws_.freeze_panes = 'C2'


def long_sheet(title, records, extra_cols=()):
    """
    1 行 = 1 モデル × 1 電流。records: list of (タグ, 電流の行 dict)
    extra_cols: (見出し, 取り出す関数, 表示形式) のタプル
    """
    ws_ = wb.create_sheet(title)
    cols = ['モデル（タグ）', '電流', '正解の\nスパイク数', '予測の\nスパイク数', '一致率', 'Γ', 'RMSE [mV]', '発散']
    cols += [c[0] for c in extra_cols]
    header(ws_, cols, [24, 30, 10, 10, 9, 9, 10, 7] + [12] * len(extra_cols))
    for i, (tag, r) in enumerate(records, 2):
        vals = [tag, r['name'], r['spikes'][0], r['spikes'][1], r['match'], r['gamma'], r['rmse'],
                '発散' if r.get('diverged') else '']
        vals += [f(r) for _, f, _ in extra_cols]
        for j, v in enumerate(vals, 1):
            c = ws_.cell(i, j, v)
            c.font = font
            if j in (5, 6):
                c.number_format = '0.000'
            elif j == 7:
                c.number_format = '0.00'
            elif j > 8:
                c.number_format = extra_cols[j - 9][2]
    last_ = len(records) + 1
    ws_.auto_filter.ref = f'A1:{get_column_letter(len(cols))}{last_}'
    ws_.conditional_formatting.add(f'F2:F{last_}', ColorScaleRule(start_type='num', start_value=0, start_color='F8696B',
                                                                  mid_type='num', mid_value=0.8, mid_color='FFEB84',
                                                                  end_type='num', end_value=1, end_color='63BE7B'))
    return ws_


# テスト 26 本（全モデル）
G_TAGS = tags_in(GAMMA_DIR)
recs = [(t, r) for t in G_TAGS for r in json.load(open(os.path.join(GAMMA_DIR, t + '.json'), encoding='utf-8'))['rows']]
long_sheet('テスト26本_電流ごと', recs)

# テスト 26 本の Γ を モデル × 電流 の表に
ws_ = wb.create_sheet('テスト26本_Γ表')
rows_g = {t: json.load(open(os.path.join(GAMMA_DIR, t + '.json'), encoding='utf-8'))['rows'] for t in G_TAGS}
names = [r['name'] for r in rows_g[G_TAGS[0]]]
header(ws_, ['モデル（タグ）', '平均 Γ'] + names, [24, 9] + [7] * len(names))
ws_.row_dimensions[1].height = 110
for j in range(3, len(names) + 3):
    ws_.cell(1, j).alignment = Alignment(text_rotation=90, horizontal='center', vertical='bottom', wrap_text=True)
ws_.cell(2, 1, '正解のスパイク数').font = Font(name='Arial', size=9, italic=True)
for j, r in enumerate(rows_g[G_TAGS[0]], 3):
    ws_.cell(2, j, r['spikes'][0]).font = Font(name='Arial', size=9, italic=True)
for i, t in enumerate(G_TAGS, 3):
    ws_.cell(i, 1, t).font = font
    c = ws_.cell(i, 2, float(sum(r['gamma'] for r in rows_g[t]) / len(rows_g[t])))
    c.font = font
    c.number_format = '0.000'
    for j, r in enumerate(rows_g[t], 3):
        c = ws_.cell(i, j, r['gamma'])
        c.font = Font(name='Arial', size=9)
        c.number_format = '0.00'
lastg = len(G_TAGS) + 2
ws_.conditional_formatting.add(f'B3:{get_column_letter(len(names) + 2)}{lastg}',
                               ColorScaleRule(start_type='num', start_value=0, start_color='F8696B',
                                              mid_type='num', mid_value=0.8, mid_color='FFEB84',
                                              end_type='num', end_value=1, end_color='63BE7B'))

# 追加テスト（上位 8 モデル）
E_TAGS = tags_in(EXT_DIR)
ext_all = {t: json.load(open(os.path.join(EXT_DIR, t + '.json'), encoding='utf-8')) for t in E_TAGS}
for suite, title, extra in [
    ('seeds', '追加_seed10本ずつ', ()),
    ('fi', '追加_f-I', (('正解の発火率\n[Hz]', lambda r: r['rate'][0], '0'), ('予測の発火率\n[Hz]', lambda r: r['rate'][1], '0'))),
    ('long', '追加_長時間20s', (('Γ 前半 5 s', lambda r: r['gamma_first5s'], '0.000'),
                                ('Γ 後半 5 s', lambda r: r['gamma_last5s'], '0.000'))),
    ('ood', '追加_分布の外', ()),
]:
    recs = [(t, r) for t in E_TAGS if suite in ext_all[t] for r in ext_all[t][suite]['rows']]
    long_sheet(title, recs, extra)

# シートの説明
ws_ = wb.create_sheet('シートの説明')
notes = [
    ('モデル一覧', 'モデルごとの要約（構造・学習方法・テスト 26 本の成績・追加テストの平均・備考）'),
    ('テスト26本_電流ごと', '全モデル × テスト 26 本の電流ごとの結果（スパイク数・一致率・Γ・RMSE）。results/gamma/<タグ>.json'),
    ('テスト26本_Γ表', '上と同じ Γ を モデル × 電流 の表にしたもの。2 行目は正解のスパイク数'),
    ('追加_seed10本ずつ', 'OU 3 種・ランダムステップ・ランダムパルスを seed 10 本ずつ（50 本、各 500 ms）。上位 8 モデル'),
    ('追加_f-I', '定常電流 0〜30 µA/cm² を 0.5 刻み（61 本、各 1 s）。発火率は後半 500 ms のスパイク数から'),
    ('追加_長時間20s', 'OU 4 本と段階 + OU 2 本を各 20 s。前半 5 s と後半 5 s の Γ も'),
    ('追加_分布の外', '強い抑制 −20、強い入力 60、チャープ 0.5→200 Hz、とても遅いランプ、下りランプ、速い OU（τ 0.1 ms）'),
    ('共通', '一致率 = 対応できたスパイク数 / max(正解の数, 予測の数)、Γ = coincidence factor（Kistler et al. 1997; '
             'Jolivet et al. 2008）、どちらも許容幅 ±2 ms。両方 0 発の電流は 1。検証 17 本の電流ごとの値は保存していない'),
]
header(ws_, ['シート', '中身'], [22, 120])
ws_.freeze_panes = None
for i, (a_, b_) in enumerate(notes, 2):
    ws_.cell(i, 1, a_).font = font
    c = ws_.cell(i, 2, b_)
    c.font = font
    c.alignment = Alignment(wrap_text=True, vertical='top')

wb.save(OUT)
print('saved', OUT)
