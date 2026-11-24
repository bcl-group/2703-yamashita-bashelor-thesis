"""
4 モデルの生データ（開発用テスト 26 本）を Markdown にする。
    python -X utf8 results/slides_1125/make_raw_md.py
出力: results/slides_1125/生データ_4モデル.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = {'A': ('hh_improve/r8_mlp_hd_w256', 'FFNN'), 'B': ('hh_improve/sp_r_init', 'RNN'),
       'C': ('hh_improve/m3_hdf_s0', 'FFNN 3 個'), 'D': ('hh_rnn3/p_h64_cl1', 'RNN 3 個')}
R = {k: json.load(open(os.path.join(ROOT, 'results', p + '.json'), encoding='utf-8'))['test'] for k, (p, _) in SRC.items()}
rows = R['A']['rows']


def cell(r):
    # 全部一致ならスパイク数だけ。外したら太字で「本数（Γ）」
    if r['gamma'] >= 0.999:
        return str(r['spikes'][1])
    return f"**{r['spikes'][1]}**（Γ {r['gamma']:.2f}）"


L = []
L.append('# 4 モデルの生データ（開発用テスト 26 本）\n')
L.append('11/25 の発表の 9・10 枚目の元データ。各モデルの学習の結果ファイル（`results/<フォルダ>/<タグ>.json` の `test`）から'
         '`results/slides_1125/make_raw_md.py` で作った。Excel の「生データ一覧」と同じ値。\n')
L.append('## 4 モデル\n')
L.append('| 分野 | 構成 | タグ | 平均 Γ | 全スパイク一致の本数 |')
L.append('|---|---|---|---:|---:|')
for k, (p, name) in SRC.items():
    L.append(f"| {k} | {name} | `{p.split('/')[1]}` | {R[k]['gamma']:.3f} | {R[k]['n100']} / 26 |")
L.append('')
L.append('## 表の読み方\n')
L.append('- 1 行が 1 本のテスト電流。**HH** の列は正解のスパイク数、**A〜D** の列は各モデルが閉ループで出したスパイク数')
L.append('- 数字だけのセルは、すべてのスパイクが HH と ±2 ms 以内で一致した電流（Γ = 1）')
L.append('- **太字（Γ）** のセルは外した電流。括弧の中がその電流の Γ（1 = 完全一致、0 = 偶然と同じ、負 = 偶然より悪い）')
L.append('- 本数が同じでも外すことがある。例：B の ramp 0-30 は 17 本どうしだが、全部が約 3 ms 早いので Γ = −0.16')
L.append('- 平均 Γ は 26 本の Γ の平均。Γ がほぼ 0 の電流が 1 本あると、平均は約 1/26 ≈ 0.04 下がる\n')
L.append('## 電流ごとの結果\n')
L.append('| 電流 | HH | A | B | C | D |')
L.append('|---|---:|---:|---:|---:|---:|')
for i, r in enumerate(rows):
    L.append(f"| {r['name']} | {r['spikes'][0]} | " + ' | '.join(cell(R[k]['rows'][i]) for k in 'ABCD') + ' |')
L.append('')
L.append('![4 モデルの生データ（表を図にしたもの）](fig_raw26.png)\n')
L.append('## 外した電流の中身\n')
L.append('灰色の太線が HH、色の線が外したモデル（A 青、B 橙、C 緑、D 黄）。\n')
L.append('![外した電流の波形](fig_fails.png)\n')
L.append('| 電流 | 何が起きたか（スパイク時刻 [ms]） |')
L.append('|---|---|')
L.append('| ramp 0→20 | HH は 492.9 ms に 1 回だけ発火。A, B, D は発火しない。C は 499.9 ms（7 ms 遅れ） |')
L.append('| ramp 0→30 | B は 17 回とも約 3 ms 早い（HH 323.5 に対して 320.5）。C は一致、D は 368.3 ms まで発火しない |')
L.append('| sin 10 Hz | 2 周期目の最初のスパイクが HH 108.9、C 118.8、D 117.3。C はこの後 4 本足りなくなる |')
L.append('| OU 5±3 | HH が発火しない 38 ms 付近で、A（37.9）と D（36.4）が余分に 1 回発火 |')
L.append('')
L.append('## 気をつけること\n')
L.append('- **RMSE だけで良し悪しは決められない**。A の const 7 は 18 本すべて一致（Γ = 1）だが、RMSE は 14.6 mV ある。'
         'スパイクは幅が約 1 ms と細いので、1 ms ずれるだけで波形の差が大きくなる')
L.append('- 26 本は作り方を決めるのに使った開発用のデータ。最終の比較には最終テスト 54 本を使う（未実施）')
L.append('- どのモデルも学習の seed は 1 本なので、seed によるばらつきはまだ見ていない')
open(os.path.join(HERE, '生データ_4モデル.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('saved')
