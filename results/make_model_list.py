# HH 代理モデルの一覧を Excel にまとめる（results/HHモデル一覧.xlsx）
# 使い方（リポジトリのルートで）: uv run python -X utf8 results/make_model_list.py
import os

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HHモデル一覧.xlsx')
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
    ('RNN 3 個（直列）', 'hh_serial（s / s_i / s_g / s_ig）',
     'RNN1→RNN2→RNN3 を同じ時刻で直列。RNN3 は g(t+dt) を入力。I を RNN1 に入れる版、G も出す版あり',
     '教師強制 → 閉ループ微調整 K=1→5→20 ms', None, None, None, '実行中',
     'ホワイトボードのやつ（別チャットの図）', 'results/hh_serial/'),
    ('RNN 3 個（直列）', 'hh_serial（p）', '比較用: 同じコードで RNN3 に g(t) を入れる（従来の並列）',
     '同上', None, None, None, '実行中', 'ホワイトボードのやつの比較用', 'results/hh_serial/'),
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

HEAD = ['No', '分類', 'モデル（タグ）', '構造', '学習方法', '検証 17 本\n100% 一致', 'テスト 26 本\n100% 一致',
        'テスト\n一致した割合', 'テスト\n平均一致率', 'テスト\n順位', 'テストで外れた電流', '備考', '結果ファイル']
WIDTH = [5, 14, 26, 46, 40, 11, 11, 11, 10, 8, 40, 44, 36]
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
            '「—」は未評価。数値は実験ログ（results/hh_improve/logs など）から転記')
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

first, last = r0 + 1, r0 + len(ROWS)
for i, row in enumerate(ROWS):
    r = first + i
    cat, tag, arch, train, val, test, mean, fails, note, path = row
    vals = [i + 1, cat, tag, arch, train, val, test, None, mean, None, fails, note, path]
    for j, v in enumerate(vals, 1):
        c = ws.cell(r, j, v if v is not None else None)
        c.font = font
        c.border = border
        c.fill = PatternFill('solid', fgColor=FILL[cat])
        c.alignment = Alignment(vertical='top', wrap_text=True,
                                horizontal='center' if j in (1, 6, 7, 8, 9, 10) else 'left')
    if val is None:
        ws.cell(r, 6, '—')
    if test is None:
        ws.cell(r, 7, '—')
        ws.cell(r, 8, '—')
        ws.cell(r, 10, '—')
    else:
        ws.cell(r, 8, f'=G{r}/{N_TEST}').number_format = '0%'
        ws.cell(r, 10, f'=IF(ISNUMBER(G{r}),RANK(G{r},$G${first}:$G${last},0),"—")')
    if mean is None:
        ws.cell(r, 9, '—')
    else:
        ws.cell(r, 9).number_format = '0.000'
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
ws.cell(lr + 1, 4, 'テスト順位は 100% 一致の本数の多い順（同数は同順位）。一致した割合 = 100% 一致の本数 / 26').font = font

wb.save(OUT)
print('saved', OUT)
